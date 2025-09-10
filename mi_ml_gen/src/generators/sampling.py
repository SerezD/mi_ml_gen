import torch

import numpy as np
from scipy.stats import truncnorm

from src.generators.utils import decode


def get_device_specific_seeds(global_rank: int, current_epoch: int):
    """
    generate random seed depending on the local machine.
    Done in order to allow different generations on each gpu!
    """

    seeds = (np.random.randint(2 ** 10, 2 ** 24, 3) + global_rank) // (current_epoch + 1)
    seeds = {'anchor': seeds[0], 'positive': seeds[1], 'discriminator': seeds[2]}

    return seeds


def truncated_noise_sample(batch_size: int, dim_z: int, mean: float, std: float, truncation: float, seed: int = None):

    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-truncation, truncation, loc=mean, scale=std, size=(batch_size, dim_z), 
                           random_state=state).astype(np.float32)
    return values


def generate_anchors(generator: torch.nn, n_samples: int, gauss_params: tuple, seed: int = None,
                     only_z: bool = False):
    """
    :param generator: bigbigan or stylegan pre_loaded model
    :param n_samples: number of images to sample
    :param gauss_params: (mean, std, truncation) parameters for sampling.
    :param seed: random seed used for sampling
    :param only_z: if True returns only the z latents

    :return z_anchors as numpy.array - images as torch tensors
    """

    mean, std, trunc = gauss_params
    dim_z = generator.dim_z

    z_anchors = truncated_noise_sample(n_samples, dim_z, mean, std, trunc, seed)

    if only_z:
        return z_anchors, None

    images = decode(generator, z_anchors)

    return z_anchors, images
