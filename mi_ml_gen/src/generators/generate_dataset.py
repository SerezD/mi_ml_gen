import argparse
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.generators.utils import load
from src.generators.sampling import generate_anchors


def parse_option():

    parser = argparse.ArgumentParser('Generate some images with the specified generator and options.'
                                     'This script has been used to generate images for the SimCLR encoder, '
                                     'since in that case we followed the experimental procedure of previous work '
                                     '(no continuous sampling).')

    parser.add_argument('--generator', type=str, choices=['bigbigan', 'stylegan'], help='which generator to use')
    parser.add_argument('--g_path', type=str, required=True, help='generator path')

    parser.add_argument('--n_images', type=int, help='total number of images to sample')
    parser.add_argument('--batch_size', type=int, help='parallel generations')
    parser.add_argument('--starting_seed', type=int, default=0, help='seed is incremented at each batch')

    # parameters for anchor sampling
    parser.add_argument('--mean', type=float, default=0., help='mean for anchor sampling')
    parser.add_argument('--std', type=float, default=1., help='std for anchor sampling')
    parser.add_argument('--truncation', type=float, default=2., help='truncation for anchor sampling')

    parser.add_argument('--save_folder', type=str, help='saving folder for images')

    opt = parser.parse_args()

    # create folder if not exists
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main(opt):

    model = load(opt.generator, opt.g_path)
    params = (opt.mean, opt.std, opt.truncation)

    for batch_idx, _ in enumerate(tqdm(range(0, opt.n_images, opt.batch_size))):

        with torch.no_grad():
            _, batch = generate_anchors(model, opt.batch_size, params, opt.starting_seed + batch_idx)

        batch = ((batch.permute(0, 2, 3, 1).cpu().numpy()) * 255.).astype(np.uint8)

        # save to file
        for j, s in enumerate(batch):
            index = (batch_idx * opt.batch_size) + j
            n = len(str(opt.n_images))
            Image.fromarray(s).save(f'{opt.save_folder}/{index:0{n}}.png')


if __name__ == '__main__':

    main(parse_option())
