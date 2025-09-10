import torch
import numpy as np

from data.loading import preprocess_stylegan_batch
from src.generators.bigbigan import make_bigbigan
from src.generators.stylegan2_ada import legacy


def disable_train(self, mode: bool = True):
    """
    Prevent PL from setting Training Mode on the Generator.
    """
    return self


def load(g_type: str, checkpoint_path: str, device: str = None):
    """
    Load and returns the pre-trained generator.
    The returned generator is in eval mode and on the selected device (defaults to cpu)
    :param g_type: one of 'bigbigan', 'stylegan'
    :param checkpoint_path: path to the pretrained model
    :param device: defaults to "cpu"
    """

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    if g_type == 'bigbigan':

        generator = make_bigbigan(checkpoint_path).to(device).eval()
        generator.device = generator.big_gan.linear.weight.device

    elif g_type == 'stylegan':

        with open(checkpoint_path, 'rb') as f:
            generator = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
        generator.dim_z = generator.z_dim  # copy attribute to get same as bigbigan
        generator.device = generator.synthesis.b4.const.device
        generator.mapping.device = generator.device
        generator.synthesis.device = generator.device

    else:
        raise NotImplementedError(f'Cannot load generator: {g_type}')

    # turn of training (for PL)
    for p in generator.parameters():
        p.requires_grad = False
    generator.train = disable_train

    # add useful attributes
    generator.type = g_type

    return generator


def to_device(generator: torch.nn.Module, device: str):

    if generator.type == 'bigbigan':
        generator.to(device)
        generator.device = device
    else:
        generator.to(device)
        generator.device = device
        generator.mapping.device = device
        generator.mapping.to(device)
        generator.synthesis.device = device
        generator.synthesis.to(device)

    return generator


def z_to_w(mapping_net: torch.nn.Module, z_vec: np.ndarray, device: str = None, truncation_psi: float = 1.):
    """
    For stylegan model
    """

    device = mapping_net.device if device is None else device

    if isinstance(z_vec, np.ndarray):
        z_vec = torch.from_numpy(z_vec).to(device)

    w = mapping_net(z_vec, None)
    w_avg = mapping_net.w_avg
    w = w_avg + (w - w_avg) * truncation_psi

    return w


def decode(model: torch.nn.Module, z_or_w_vec: torch.Tensor | np.ndarray):
    """
    from z_vector or w_vector to generated images through the passed model
    The returned images are in 0 __ 1 range
    """

    device = model.device

    if isinstance(z_or_w_vec, np.ndarray):
        z_or_w_vec = torch.tensor(z_or_w_vec, device=device)

    if len(z_or_w_vec.shape) == 2:
        z_vec, w_vec = z_or_w_vec, None
    else:
        z_vec, w_vec = None, z_or_w_vec

    if model.type == 'bigbigan':

        images = model(z_vec.to(device))

        # denormalize
        images = torch.clip((images + 1) / 2., 0., 1.)

    elif model.type == 'stylegan':

        if w_vec is None:
            w_vec = z_to_w(model.mapping, z_vec, device)

        images = model.synthesis(w_vec, noise_mode='const')

        # reformat to 128 x 128
        images = preprocess_stylegan_batch(images)

        # denormalize
        images = (images * 127.5 + 128).clamp(0, 255) / 255.

    else:

        raise NotImplementedError(f'unknown generator passed: {model.type}')

    return images
