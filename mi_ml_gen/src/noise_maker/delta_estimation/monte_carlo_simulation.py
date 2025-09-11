import argparse

import numpy as np
from glob import glob

import torch
from einops import rearrange
from tqdm import tqdm

from src.generators.sampling import truncated_noise_sample
from src.generators.utils import load, z_to_w
from src.noise_maker.walkers import mse_dist


def init_run():

    parser = argparse.ArgumentParser('estimate delta values on each chunk of generator')

    parser.add_argument('--walkers_base_path', type=str, required=True,
                        help='path to walkers to test for each chunk. Must contain n_chunks subfolders names '
                             '"chunk{i}", each containing a checkpoint named w_COPGen_{iter_name}.pth')

    parser.add_argument('--walkers_iter', nargs='+',
                        help='contains n_chunks string elements defining each {iter_name} for loading '
                             'a specific checkpoint')

    parser.add_argument('--n_samples', type=int, default=500000, help='number of samples for the estimation')

    # tz parameters
    parser.add_argument('--mean', type=float, default=0., help='mean for anchor sampling')
    parser.add_argument('--std', type=float, default=1., help='std for anchor sampling')
    parser.add_argument('--trunc', type=float, default=2., help='truncation for anchor sampling')

    # generator specific_parameters
    parser.add_argument('--g_path', type=str, default=None, help='needed only for StyleGan')
    parser.add_argument('--dim_z', type=int, default=120, help='dimension of the whole z vector')

    opt = parser.parse_args()

    return opt


def format_table(deltas: np.ndarray, loss_values: list):

    print('chunk & mean & std & INFO NCE \\\\')

    for chunk in range(deltas.shape[0]):

        this_deltas = deltas[chunk]
        mu = np.mean(this_deltas)
        std = np.std(this_deltas)
        print(f'{chunk} & {mu:.2f} & {std:.2f} & {loss_values[chunk]:.2f} \\\\')


def main(opt: argparse.Namespace):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    div = 100
    bs = opt.n_samples // div

    # load generator
    if opt.g_path is not None:
        mapping_net = load('stylegan', opt.g_path, device).mapping
    else:
        mapping_net = None

    deltas_estimation = np.empty((0, opt.n_samples))
    infonce_values = []

    walker_base_paths = sorted(glob(f'{opt.walkers_base_path}/*/'))

    for i, (base_path, iter_name) in enumerate(zip(walker_base_paths, opt.walkers_iter)):

        print(f'[INFO] Measuring chunk {i + 1} / {len(walker_base_paths)}')

        # load walker
        walker = torch.load(f'{base_path}/w_COPGen_{iter_name}.pth')['nn_walker']
        walker.device_ids = [0]

        # initialize delta i
        delta_i = np.empty((1, 0))

        for _ in tqdm(range(div)):

            # generate anchors
            anchors = truncated_noise_sample(bs, dim_z=opt.dim_z, mean=opt.mean, std=opt.std, truncation=opt.trunc)

            # generate positives for walker and w anchors if StyleGan
            with torch.no_grad():
                
                _, positives, _ = walker(torch.tensor(anchors, device=device), mapping_net=mapping_net)
                positives = positives.cpu().numpy()
                
                if mapping_net is not None:
                    anchors = z_to_w(mapping_net, anchors, device=device).cpu().numpy() 

            # estimate deltas
            if len(anchors.shape) != 3:
                # BigGan Case
                anchors = rearrange(anchors, 'b (n d) -> b n d', d=walker.z_chunk_dim)
                positives = rearrange(positives, 'b (n d) -> b n d', d=walker.z_chunk_dim)
            else:
                # StyleGan --> Regroup
                anchors = rearrange(anchors, 'b (g n) d -> b g (n d)', g=4)
                positives = rearrange(positives, 'b (g n) d -> b g (n d)', g=4)

            estimated_deltas = mse_dist(anchors[:, i], positives[:, i])
            delta_i = np.concatenate([delta_i, np.expand_dims(estimated_deltas, 0)], axis=1)

        deltas_estimation = np.concatenate([deltas_estimation, delta_i], axis=0)

        # Load Loss Values
        loss_values = np.load(f'{base_path}/F_loss_values.npy')
        n_losses = len(loss_values)
        n_ckpts = len(glob(f'{base_path}/*.pth'))
        idx = (n_losses // n_ckpts) * (int(iter_name) // 2048)
        loss_value = np.mean(loss_values[idx - 1: idx + 2])  # remove a little the oscillations
        infonce_values.append(loss_value)

    format_table(deltas_estimation, infonce_values)


if __name__ == '__main__':

    main(init_run())
