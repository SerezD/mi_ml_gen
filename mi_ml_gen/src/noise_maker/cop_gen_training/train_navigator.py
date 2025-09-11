import os
import argparse
import warnings

import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as ddp

from einops import rearrange

from PIL import Image
from kornia.augmentation import AugmentationSequential, RandomResizedCrop, RandomHorizontalFlip, ColorJiggle, \
    RandomGrayscale, CenterCrop, Normalize
from torchvision.utils import make_grid

from src.noise_maker.walkers import NonlinearWalker
from src.noise_maker.delta_estimation.visualize_walker import plot_loss
from src.noise_maker.cop_gen_training.networks.resnet_big import SupConResNet
from src.noise_maker.cop_gen_training.losses import SupConLoss
from src.generators.sampling import truncated_noise_sample
from src.generators.utils import load, decode


def init_run():
    
    desc = ('Complete refactor of cop_gen walker training: '
            'https://github.com/LiYinqi/COP-Gen/blob/master/cop_gen/train_navigator_bigbigan_optimal.py'
            'We include training for StyleGan generator and DDP training.'
            )
    
    parser = argparse.ArgumentParser(desc)

    parser.add_argument('--generator', type=str, choices=['bigbigan', 'stylegan'], help='which generator to use')
    parser.add_argument('--g_path', type=str, required=True, help='generator path')

    parser.add_argument('--chunks', type=str, help='start and end chunk index, formatted as "start_end", starting'
                                                   'from 0, first included and last excluded. E.g. "0_6" means chunks '
                                                   'from 0 to 5 (all the chunks in bigbigan)')

    parser.add_argument('--num_samples', type=int, default=1024000, help='number of training samples')

    parser.add_argument('--save_freq', type=int, default=2048, help='frequency (num_samples) to save weights')
    parser.add_argument('--save_folder', type=str, default='./walker_runs', help='saving folder')

    # tz parameters
    parser.add_argument('--mean', type=float, default=0., help='mean for anchor sampling')
    parser.add_argument('--std', type=float, default=1., help='std for anchor sampling')
    parser.add_argument('--truncation', type=float, default=2., help='truncation for anchor sampling')

    # data space transformation Tx (in practice not used, but kept for consisency)
    parser.add_argument('--simclr_aug', action='store_true', help='use full data augs')
    parser.add_argument('--removeCrop', action='store_true', help='remove random crop from data augs')
    parser.add_argument('--removeColor', action='store_true', help='remove random color from data augs')

    parser.add_argument('--run_name', type=str, default='0', help='use to track run name')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_walk', type=float, default=8e-5, help='lr for non linear walker')
    parser.add_argument('--lr_MI', type=float, default=5e-5, help='lr for F encoder')

    # early stopping
    parser.add_argument('--early_stopping_min_steps', type=int, default=100000,
                        help='min number of steps that walker will perform')

    parser.add_argument('--early_stopping_threshold', type=float, default=2.5,
                        help='when F loss goes back above this threshold, stops training')

    # fixed arguments
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--temp', type=float, default=0.1)

    opt = parser.parse_args()

    desc = f'{opt.run_name}_generator={opt.generator}_chunks={opt.chunks}'

    if opt.simclr_aug:
        desc += '_simclrAug'
    elif opt.removeCrop:
        desc += '_removeCrop'
    elif opt.removeColor:
        desc += '_removeColor'
    else:
        desc += '_noAug'

    # create checkpoint folder
    opt.ckpt_folder = os.path.join(opt.save_folder, 'ckpts', desc)

    if not os.path.isdir(opt.ckpt_folder):
        os.makedirs(opt.ckpt_folder)

    return opt


def setup(rank, world_size):

    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        if rank == 0:
            warnings.warn("Set Environ Variable 'MASTER_ADDR'='localhost'")

    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
        if rank == 0:
            warnings.warn("Set Environ Variable 'MASTER_PORT'='29500'")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def set_models(opt):

    setup(opt.rank, opt.world_size)

    # Generator
    generator = load(opt.generator, opt.g_path, device=opt.rank)

    # Walker
    nn_walker = NonlinearWalker(g_type=opt.generator, chunks=opt.chunks).to(opt.rank)

    # mutual info estimator
    f = SupConResNet(name='resnet18', img_size=int(opt.img_size)).to(opt.rank)

    nn_walker = ddp(nn_walker, device_ids=[opt.rank], static_graph=True, gradient_as_bucket_view=True)
    f.encoder = ddp(f.encoder, device_ids=[opt.rank], static_graph=True, gradient_as_bucket_view=True)

    return generator, nn_walker, f


class Tx:
    """
    Rewritten to use kornia transforms
    """
    def __init__(self, opt):

        crop_size = int(opt.img_size * 0.875)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if opt.simclr_aug:
            self.img_transform = AugmentationSequential(
                RandomResizedCrop((crop_size, crop_size), scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                ColorJiggle(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                Normalize(mean=mean, std=std),
                same_on_batch=False)

        elif opt.removeCrop:
            self.img_transform = AugmentationSequential(
                CenterCrop((crop_size, crop_size)),
                RandomHorizontalFlip(),
                ColorJiggle(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                Normalize(mean=mean, std=std),
                same_on_batch=False)

        elif opt.removeColor:
            self.img_transform = AugmentationSequential(
                RandomResizedCrop((crop_size, crop_size), scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                Normalize(mean=mean, std=std),
                same_on_batch=False)
        else:
            self.img_transform = AugmentationSequential(
                CenterCrop((crop_size, crop_size)),
                Normalize(mean=mean, std=std),
                same_on_batch=False)

    def apply(self, imgs):
        return self.img_transform(imgs)


def train(rank, world_size, opt):

    torch.cuda.set_device(rank)

    opt.rank = rank
    opt.world_size = world_size

    generator, walker, f_resnet = set_models(opt)

    tx = Tx(opt)

    # criterion and optimizer
    criterion = SupConLoss(temperature=opt.temp).to('cuda')

    optimizer_walk = optim.Adam(walker.parameters(), lr=opt.lr_walk,
                                betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    optimizer_mi = optim.Adam(f_resnet.parameters(), lr=opt.lr_MI,
                              betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    cudnn.benchmark = True

    opt.save_point = opt.save_freq

    # save images
    img_save_path = f'{opt.ckpt_folder}/images/'
    if opt.rank == 0 and not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    if opt.rank == 0:
        loss_f_values = []
        delta_values = []

    # train loop
    for batch_idx in range(opt.num_samples // (opt.batch_size * world_size)):

        # sample z randomly
        z = truncated_noise_sample(batch_size=opt.batch_size, dim_z=generator.dim_z, mean=opt.mean, std=opt.std,
                                   truncation=opt.truncation)
        z = torch.from_numpy(z).to('cuda')

        # z' = z + Tz(z)
        mapping_net = generator.mapping if opt.generator == 'stylegan' else None
        anchor, positive, estimated_delta = walker(z, mapping_net=mapping_net)

        # generate anchor and positive image
        if opt.generator == 'bigbigan':
            img_anchor = decode(generator, anchor)
            img_positive = decode(generator, positive)
        else:
            img_anchor = decode(generator, anchor)
            img_positive = decode(generator, positive)

        # data space transformations tx
        img_anchor = tx.apply(img_anchor)
        img_positive = tx.apply(img_positive)

        # cat anchors and postives
        images = rearrange(torch.stack([img_anchor, img_positive], dim=1), 'b n c h w -> (b n) c h w')

        # #### Update mutual_info_estimator: min InfoNCE loss (max lower bound of Mutual Info) ######
        optimizer_mi.zero_grad()

        features = f_resnet(images.detach())
        features = features.view(opt.batch_size, 2, -1)

        loss_f = criterion(features)

        loss_f.backward()
        optimizer_mi.step()
        #############################################################################################

        # #### Update navigator (nn_walker): max InfoNCE loss (min Mutual Info) #####################
        optimizer_walk.zero_grad()

        features = f_resnet(images)
        features = features.view(opt.batch_size, 2, -1)

        loss_walker = - criterion(features)

        loss_walker.backward()
        optimizer_walk.step()
        #############################################################################################

        # gather between all ranks
        this_values = torch.tensor(np.array([np.mean(estimated_delta), loss_f.detach().cpu().numpy()]), device='cuda')
        if opt.rank == 0:
            collected = [torch.zeros_like(this_values) for _ in range(world_size)]
            dist.gather(gather_list=collected, tensor=this_values)
        else:
            dist.gather(tensor=this_values, dst=0)

        # update lists on rank 0
        n_samples = batch_idx * opt.batch_size * world_size

        if opt.rank == 0:
            collected = torch.stack(collected)
            loss_f_values.append(torch.mean(collected[:, 1]).cpu().numpy())
            delta_values.append(torch.mean(collected[:, 0]).cpu().numpy())
            print(f'[INFO] trained_samples:{n_samples}, Loss F: {loss_f_values[-1]}, Delta: {delta_values[-1]}')

        # save anchor-positive pairs for monitoring the minimax training process
        if opt.rank == 0 and (n_samples >= opt.save_point) and (batch_idx > 0):

            max_samples = 8
            grid_images = make_grid(torch.cat((img_anchor[:max_samples], img_positive[:max_samples].detach())),
                                    nrow=max_samples, normalize=True)
            grid_images = (grid_images.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(grid_images).save(f'{img_save_path}step={n_samples}.png', 'PNG')

            print('Saving navigator')
            torch.save({'nn_walker': walker.module},
                       os.path.join(opt.ckpt_folder, f'w_COPGen_{n_samples}.pth'))

            opt.save_point += opt.save_freq

        # early stopping
        if opt.rank == 0 and n_samples > opt.early_stopping_min_steps and \
                loss_f_values[-1] > opt.early_stopping_threshold:
            early_stop = torch.tensor(1, device=opt.rank)
        else:
            early_stop = torch.tensor(0, device=opt.rank)

        # send signal to all processes
        dist.broadcast(early_stop, src=0, group=dist.group.WORLD)
        if bool(early_stop):
            print(f'rank {opt.rank} - early stopping')
            break

    if opt.rank == 0:
        np.save(os.path.join(opt.ckpt_folder, 'F_loss_values.npy'), loss_f_values)
        np.save(os.path.join(opt.ckpt_folder, 'delta_values.npy'), delta_values)

        plot_loss(opt.batch_size * opt.world_size, img_save_path, loss_f_values, delta_values)

    dist.destroy_process_group()


if __name__ == '__main__':

    w_size = torch.cuda.device_count()
    print(f'Using {w_size} gpus for training model')

    mp.spawn(train, args=(w_size, init_run()), nprocs=w_size)
