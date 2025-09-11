"""
Utils for plotting
"""
import os.path
from glob import glob

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from src.generators.sampling import truncated_noise_sample
from src.generators.utils import load, decode

# args
# PATH = '~/walkers/bigbigan_all_chunks/ours/'
# walker_ckpts = glob(f'{PATH}*.pth')
# generator_path = '~/runs/bigbigan/BigBiGAN_x1.pth'
# training_bs = 64
# generator_type = 'bigbigan'
# mean, std, trunc = 0., 1., 2.

PATH = '~/walkers/stylegan_grouped_chunks/chunks=12_16/'
walker_ckpts = glob(f'{PATH}*.pth')
generator_path = '~/runs/stylegan2/stylegan2-car-config-f.pkl'
training_bs = 64
generator_type = 'stylegan'
mean, std, trunc = 0., 0.9, 1.


def plot_loss(batch_size: str, path: str, f_loss_values: np.ndarray, delta_estimation: np.ndarray):

    # display 10 x ticks (no matter how many values)
    n_ticks = 10
    xticks = np.arange(0, len(f_loss_values) + 1, len(f_loss_values) / n_ticks)
    xticks_labels = [f'{int(xt * batch_size) / 1000:.2f} K' for xt in xticks]

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(f_loss_values)
    ax[0].set_title('MI Estimator (f) loss', loc='left')
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(xticks_labels)
    ax[0].grid(True)
    ax[0].set_xlabel('Training samples', loc='right')

    ax[1].plot(delta_estimation)
    ax[1].set_title('Delta Estimation', loc='left')
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xticks_labels)
    ax[1].grid(True)
    ax[1].set_xlabel('Training samples', loc='right')

    plt.savefig(f'{path}/loss_plot.png')
    plt.close(fig)


def plot_samples(generator, walker, save_path, n, seed):

    # 1. sample anchors and positives
    z = truncated_noise_sample(batch_size=n, dim_z=generator.dim_z, mean=mean, std=std, truncation=trunc, seed=seed)
    z = torch.from_numpy(z).to('cuda')

    # z' = z + Tz(z)
    mapping_net = None if generator_type == 'bigbigan' else generator.mapping
    anchor, positive, _ = walker(z, mapping_net=mapping_net)

    # generate anchor image
    with torch.no_grad():
        img_anchor = decode(generator, anchor)
        img_positive = decode(generator, positive)

    images = rearrange(torch.stack([img_anchor, img_positive], dim=1), 'b n c h w -> (b n) c h w')
    grid_images = make_grid(images, nrow=8)
    grid_images = (grid_images.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    plt.imshow(grid_images)
    plt.axis(False)
    plt.savefig(save_path)


def main(main_path: str, all_ckpts: list[str], g_type: str, g_path: str, batch_size: int, device: str):

    generator = load(g_type, g_path, device=device)

    if not os.path.exists(f'{main_path}images/'):
        os.makedirs(f'{main_path}images/')

    for ckpt in tqdm(all_ckpts):

        walker = torch.load(ckpt)['nn_walker']
        walker.device_ids = [0]
        walker.to(device).eval()

        step = ckpt.split('_')[-1].split('.')[0]
        save_pth = f'{main_path}images/step={step}.png'

        plot_samples(generator, walker, save_path=save_pth, n=16, seed=1)

    loss_values = np.load(f'{main_path}F_loss_values.npy')
    delta_values = np.load(f'{main_path}delta_values.npy')

    plot_loss(batch_size, f'{main_path}images/', loss_values, delta_values)


if __name__ == '__main__':

    main(main_path=PATH, all_ckpts=walker_ckpts, g_type=generator_type, g_path=generator_path, batch_size=training_bs,
         device='cuda' if torch.cuda.is_available() else 'cpu')
