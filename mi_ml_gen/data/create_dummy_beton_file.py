import argparse
import os

from ffcv_pl.generate_dataset import create_beton_wrapper

from data.datasets import FakeDataset


def get_args():

    parser = argparse.ArgumentParser(
        description='Define a Dummy Int Dataset using ffcv for fast data loading. '
                    'Used for online generation of training data. '
                    'It ensures the same epoch length between real and generated training')

    parser.add_argument('--output_folder', type=str, required=True,
                        help='folder that will contain the generated .beton file')

    parser.add_argument('--train_samples', type=int, help='number of training samples to be generated.')

    opt = parser.parse_args()

    if not os.path.exists(opt.output_folder):
        print(f'[INFO] Creating output folder: {opt.output_folder}')
        os.makedirs(opt.output_folder)

    return opt


def main(opt):

    dataset = FakeDataset(opt.train_samples)
    create_beton_wrapper(dataset, f"{opt.output_folder}/train.beton")


if __name__ == '__main__':

    main(get_args())
