import argparse
import numpy as np
import os
import torch
import yaml

from PIL import Image
from tqdm import tqdm

from src.noise_maker.walkers import MultipleRandomWalker
from src.generators.sampling import generate_anchors
from src.generators.utils import decode, load, z_to_w


def parse_option():

    parser = argparse.ArgumentParser('Generate some images and augmentations using "random" perturbations.')

    parser.add_argument('--configuration', type=str, help='absolute path to model conf file .yaml')
    parser.add_argument('--save_folder', type=str, help='saving folder for images')

    opt = parser.parse_args()

    # create folder if not exists
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_model_config(params_conf_path: str):
    """
    reads yaml file set in path and returns config corresponding to dataset
    """

    with open(params_conf_path, 'r', encoding='utf-8') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main(opt):
    
    conf = get_model_config(opt.configuration)

    model = load(conf['generator'], conf['generator_path'])
    params, starting_seed = conf['anchor_params'], conf['starting_seed']
    n_images, batch_size, augmentations = conf['n_images'], conf['batch_size'], conf['augmentations']
    device = model.device

    Tz = MultipleRandomWalker(conf['generator'], conf['Tz'])
    Tz.training = False
    
    all_images = None

    for batch_idx, _ in enumerate(tqdm(range(0, n_images, batch_size))):

        with torch.no_grad():

            z_anchors, _ = generate_anchors(model, batch_size, params, starting_seed + batch_idx, only_z=True)

            # now get positives.
            if conf['generator'] == 'bigbigan':

                latent_anchors = torch.from_numpy(z_anchors).to(device)
                latent_positives = []

                for i in range(1, augmentations + 1):
                    _, z_positives, _ = Tz(z_anchors, seed=starting_seed + batch_idx + i)
                    latent_positives.append(torch.from_numpy(z_positives).to(device))

            else:

                latent_anchors = z_to_w(model.mapping, z_anchors)
                latent_positives = []

                for i in range(1, augmentations + 1):

                    _, w_positives, _ = Tz(z_anchors, 
                                           seed=starting_seed + batch_idx + i,
                                           mapping_net=model.mapping)
                    latent_positives.append(w_positives)

            anchors = decode(model, latent_anchors)
            positives = [decode(model, p) for p in latent_positives]
            
            full_batch = torch.stack([anchors] + positives, dim=1)
            if all_images is None:
                all_images = full_batch.cpu()
            else:
                all_images = torch.cat([all_images, full_batch.cpu()], dim=0)

    # save all to file
    for b_idx, batch in enumerate(all_images):
        for i_idx, image in enumerate(batch):
            name = f'idx={b_idx}_var={i_idx}.png'
            Image.fromarray((image.permute(1, 2, 0).numpy() * 255.).astype(np.uint8)).save(f'{opt.save_folder}/{name}')


if __name__ == '__main__':
        
    main(parse_option())
