import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from data.loading import get_datamodule
from src.evaluations.classification.model import LinCls
from src.multiview_encoders.utils.system import set_matmul_precision, available_cpu_count


def init_run():
    parser = argparse.ArgumentParser(description="script for evaluating classification performance.")

    parser.add_argument('--batch_size', type=int, help='batch size to use. Note: test is performed on one gpu!')

    parser.add_argument('--dataset', type=str, help='name of the dataset to test for classification',
                        choices=['birdsnap', 'caltech101', 'cifar100', 'DTD', 'fgvc-aircraft-2013b', 
                                 'flowers102', 'food101', 'imagenet', 'imagenet100', 'pets', 'StanfordCars'])

    parser.add_argument('--lin_cls_path', type=str, 
                        help="absolute path leading to the checkpoint folder of the pretrained classifier.")

    parser.add_argument('--data_path', type=str,
                        help="path to a folder containing the 'test' beton file of the labelled dataset.")

    parser.add_argument('--out_log_file', type=str, help='out file where logs will be saved (.csv will be added)')

    args = parser.parse_args()

    return args


def main(opt):

    set_matmul_precision()

    machine_conf = {}

    machine_conf['num_nodes'] = 1
    machine_conf['gpus'] = 1
    machine_conf['rank'] = 0
    machine_conf['is_dist'] = False

    machine_conf['workers'] = min(available_cpu_count(), 16)

    # cumulative bs == local_bs
    machine_conf['cumulative_bs'] = opt.batch_size
    machine_conf['local_bs'] = opt.batch_size

    # load model
    model = LinCls.load_from_checkpoint(checkpoint_path=opt.lin_cls_path, pretrained_path=None, dataset=opt.dataset,
                                        train_conf={'lr': 0.3, 'decay': False, 'max_epochs': 1})  # dummy conf

    # init trainer
    path_split = opt.out_log_file.split('/')
    out_dir = '/'.join(path_split[:-1])
    name = f"{path_split[-1]}"

    logger = CSVLogger(save_dir=out_dir, name=name)
    trainer = pl.Trainer(strategy='ddp', accelerator='gpu', logger=logger,
                         num_nodes=machine_conf['num_nodes'], devices=machine_conf['gpus'],
                         deterministic=True)

    # data module
    test_file = f'{opt.data_path}test.beton'

    data_module = get_datamodule('bigbigan' if opt.dataset == 'imagenet' else None,
                                 machine_conf['local_bs'], machine_conf['workers'],
                                 0, machine_conf['is_dist'], False, test_file=test_file)

    print(f"conf:\n{machine_conf}")

    trainer.test(model, data_module, opt.lin_cls_path)


if __name__ == '__main__':

    main(init_run())
