import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp

from einops import rearrange

from codecarbon import OfflineEmissionsTracker
import time

from tqdm import tqdm

from src.noise_maker.cop_gen_training.networks.resnet_big import SupConResNet
from src.noise_maker.cop_gen_training.losses import SupConLoss
from src.generators.sampling import truncated_noise_sample
from src.generators.utils import load, decode
from src.test_online_learning.utils import setup, Tx


def init_run():

    parser = argparse.ArgumentParser('train a small encoder and measure the time and co2 emissions')

    parser.add_argument('--generator_path', type=str, help='generator path')

    parser.add_argument('--iters_per_epoch', type=int, default=126789, 
                        help='training iters per epoch (num images in original dataset)')

    parser.add_argument('--num_epochs', type=int, default=20, help='number of training epochs')

    opt = parser.parse_args()

    return opt


def set_models(opt):

    # Generator
    generator = load('bigbigan', opt.generator_path, device=opt.rank)

    # mutual info estimator
    f = SupConResNet(name='resnet18', img_size=112).to(opt.rank)
    f.encoder = ddp(f.encoder, device_ids=[opt.rank], static_graph=True, gradient_as_bucket_view=True)

    return generator, f


def train(rank, world_size, opt):

    torch.cuda.set_device(rank)

    opt.rank = rank
    opt.world_size = world_size

    setup(opt.rank, opt.world_size)

    # init training objects
    generator, f_resnet = set_models(opt)
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

        for _ in tqdm(range(opt.iters_per_epoch // (opt.batch_size * world_size))):

            with torch.no_grad():
                
                generator.to(opt.rank)

                # generate batch (no Tz)
                z = truncated_noise_sample(batch_size=opt.batch_size, dim_z=generator.dim_z, mean=0., std=1.,
                                           truncation=2.)
                z = torch.from_numpy(z).cuda(opt.rank)

                batch = decode(generator, z)

                generator.to('cpu')

            # data space transformations tx
            img_anchor = tx.apply(batch)
            img_positive = tx.apply(batch)

            # cat anchors and postives
            images = rearrange(torch.stack([img_anchor, img_positive], dim=1), 'b n c h w -> (b n) c h w')

            #############################################################################################
            optimizer.zero_grad()

            features = f_resnet(images)
            features = features.view(opt.batch_size, 2, -1)

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
        print(f'[INFO] Tracking online with bs = {opt.batch_size}')

        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()
        mp.spawn(train, args=(w_size, opt), nprocs=w_size)
        tracker.stop()
        print(' ######################################################### ')
