import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize

from ffcv import Loader
from ffcv.loader import OrderOption

from einops import rearrange

from codecarbon import OfflineEmissionsTracker
import time

from tqdm import tqdm

from data.loading import bigbigan_transforms
from src.noise_maker.cop_gen_training.networks.resnet_big import SupConResNet
from src.noise_maker.cop_gen_training.losses import SupConLoss
from src.test_online_learning.utils import setup, Tx


def init_run():

    parser = argparse.ArgumentParser('train a small encoder and measure epoch training time and co2 emissions')

    parser.add_argument('--loader', choices=['ffcv', 'torch'])

    parser.add_argument('--dataset_path', type=str,
                        help='path to the folder containing train/***.png or ffcv/train.beton files.')

    parser.add_argument('--num_epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--workers', type=int, default=8, help='number of parallel workers')

    opt = parser.parse_args()

    return opt


def get_loader(opt):

    if opt.loader == 'ffcv':
        loader = Loader(f'{opt.dataset_path}ffcv/train.beton',
                        batch_size=opt.batch_size,
                        num_workers=opt.workers,
                        order=OrderOption.RANDOM,
                        distributed=opt.world_size > 1,
                        os_cache=True,
                        seed=0,
                        pipelines={'image_0': bigbigan_transforms.copy()})
    else:

        transforms = Compose([ToTensor(), Resize(size=(128, 128), antialias=True)])
        train_data = ImageFolder(f'{opt.dataset_path}train/', transform=transforms)
        sampler = DistributedSampler(train_data, num_replicas=opt.world_size, rank=opt.rank)
        loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                            pin_memory=True, sampler=sampler)

    return loader


def set_model(opt):

    # mutual info estimator
    f = SupConResNet(name='resnet18', img_size=112).to(opt.rank)
    f.encoder = ddp(f.encoder, device_ids=[opt.rank], static_graph=True, gradient_as_bucket_view=True)

    return f


def train(rank, world_size, opt):

    torch.cuda.set_device(rank)

    opt.rank = rank
    opt.world_size = world_size

    setup(opt.rank, opt.world_size)

    # init training objects
    f_resnet = set_model(opt)
    loader = get_loader(opt)
    tx = Tx()
    tx.img_transform.cuda(rank)
    criterion = SupConLoss(temperature=0.1).cuda(rank)
    optimizer = optim.Adam(f_resnet.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=0.)

    epochs_gpu_time = []

    # train loop
    for epoch in range(opt.num_epochs):

        if rank == 0:
            print(f'Epoch: {epoch}')

        start_time = time.time()

        if opt.loader == 'torch':
            loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(loader)):

            batch = batch[0].cuda(rank, non_blocking=True)

            bs, _, _, _ = batch.shape

            # data space transformations tx
            img_anchor = tx.apply(batch)
            img_positive = tx.apply(batch)

            # cat anchors and postives
            images = rearrange(torch.stack([img_anchor, img_positive], dim=1), 'b n c h w -> (b n) c h w')

            #############################################################################################
            optimizer.zero_grad()

            features = f_resnet(images)
            features = features.view(bs, 2, -1)

            loss_f = criterion(features)

            loss_f.backward()
            optimizer.step()
            #############################################################################################

        if rank == 0:
            epochs_gpu_time.append((time.time() - start_time) * opt.world_size)

    if rank == 0:
        print(f'[INFO] Mean GPU Time per epoch (seconds) = {np.mean(epochs_gpu_time)}')

    dist.destroy_process_group()


if __name__ == '__main__':

    w_size = torch.cuda.device_count()
    print(f'Using {w_size} gpus for training model')

    # options
    opt = init_run()

    for batch_size in (32, 64, 128, 256):

        opt.batch_size = batch_size

        print(f'[INFO] Tracking offline with bs = {opt.batch_size} and Loader = {opt.loader}')

        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()
        mp.spawn(train, args=(w_size, opt), nprocs=w_size)
        tracker.stop()
        print(' ######################################################### ')
