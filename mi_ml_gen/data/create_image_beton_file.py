import argparse
import os

from ffcv.fields import RGBImageField
from ffcv_pl.generate_dataset import create_beton_wrapper

from data.datasets import ImageLabelDataset, ImageDataset


def get_args():

    parser = argparse.ArgumentParser(
        description='Define an Image Dataset using ffcv for fast data loading.')

    parser.add_argument('--output_folder', type=str, required=True,
                        help='folder that will contain the generated .beton files')

    parser.add_argument('--train_folder', type=str, default=None,
                        help='optional path to train data folder. The folder can contain images or '
                             'subdirectories (classes)')
    
    parser.add_argument('--val_folder', type=str, default=None,
                        help='optional path to validation data folder. The folder can contain images or '
                             'subdirectories (classes)')
    
    parser.add_argument('--test_folder', type=str, default=None,
                        help='optional path to test data folder. The folder can contain images or '
                             'subdirectories (classes)')
    
    parser.add_argument('--predict_folder', type=str, default=None,
                        help='optional path to predict data folder. The folder can contain images or '
                             'subdirectories (classes)')

    parser.add_argument('--max_resolution', type=int, default=256, 
                        help='max resolution accepted for the saved images.'
                             'used values for each dataset are:'
                             'birdsnap: 224,'
                             'caltech101: 256,'
                             'cifar100: 32,'
                             'DTD: 256,'
                             'fgvc-aircraft-2013b: 256,'
                             'flowers102: 256,'
                             'food101: 256,'
                             'imagenet: 256,'
                             'imagenet100: 256,'
                             'lsun_cars: 512,'
                             'pets: 256,'
                             'stanford_cars: 256'
                        )

    parser.add_argument('--with_labels', action='store_true',
                        help='if specified, will create Image Label dataset using subfolders for classes splits.')

    opt = parser.parse_args()

    if not os.path.exists(opt.output_folder):
        print(f'[INFO] Creating output folder: {opt.output_folder}')
        os.makedirs(opt.output_folder)

    return opt


def main(opt):

    folders = [opt.train_folder, opt.val_folder, opt.test_folder, opt.predict_folder]
    out_names = ['train', 'validation', 'test', 'predict']

    for folder, name in zip(folders, out_names):

        if folder is not None:

            print(f'[INFO] creating {name} beton file...')

            if opt.with_labels:
                dataset = ImageLabelDataset(folder=folder, image_size=opt.max_resolution)
                fields = (RGBImageField(write_mode='jpg', max_resolution=opt.max_resolution), None)
            else:
                dataset = ImageDataset(folder=folder, image_size=opt.max_resolution)
                fields = (RGBImageField(write_mode='jpg', max_resolution=opt.max_resolution), )

            create_beton_wrapper(dataset, f"{opt.output_folder}/{name}.beton", fields)

        else:
            print(f'[INFO] skipping {name} beton file...')


if __name__ == '__main__':

    main(get_args())
