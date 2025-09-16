import argparse
import pathlib

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import os
import torch
import yaml

from data.loading import get_datamodule
from src.evaluations.classification.model import LinCls
from src.multiview_encoders.utils.system import set_matmul_precision, available_cpu_count


def init_run():

    parser = argparse.ArgumentParser(description="script for multiview representation learning Encoders training.")

    parser.add_argument('--encoder_path', type=str, 
                        help="absolute path leading to the checkpoint folder of the pretrained encoder.")

    parser.add_argument('--data_path', type=str, 
                        help=("path to a folder containing the 'train' and 'validation' "
                              "beton files of labelled datasets."))

    parser.add_argument('--run_name', type=str)

    parser.add_argument('--dataset', type=str, help='name of the dataset to train for classification',
                        choices=['birdsnap', 'caltech101', 'cifar100', 'DTD', 'fgvc-aircraft-2013b', 
                                 'flowers102', 'food101', 'imagenet', 'imagenet100', 'pets', 'StanfordCars'])

    parser.add_argument('--seed', type=int, help='seed for randomness reproducibility')

    parser.add_argument('--conf', type=str, default='classifier',
                        help='name of the model conf file - without extension')

    parser.add_argument('--logging', action='store_true', help='log results to wandb')

    parser.add_argument('--resume_from', type=str, help='relative path to checkpoint to restore run', default=None)
    parser.add_argument('--wandb_id', type=str, help='wandb run id to restore', default=None)

    args = parser.parse_args()

    return args


def get_model_config(params_conf_path: str):
    """
    reads yaml file set in path and returns config corresponding to dataset
    """

    with open(params_conf_path, 'r', encoding='utf-8') as stream:
        conf = yaml.safe_load(stream)
    return conf['train']


def main(opt):

    pl.seed_everything(opt.seed, workers=True)

    set_matmul_precision()

    opt.project_path = str(pathlib.Path(__file__).parent.resolve()).split('src')[0]

    # get model config
    conf_path = f'{opt.project_path}configurations/classifiers/{opt.conf}.yaml'
    train_conf = get_model_config(conf_path)

    # config variables
    train_conf['num_nodes'] = int(os.getenv('NODES')) if os.getenv('NODES') is not None else 1
    train_conf['gpus'] = torch.cuda.device_count()
    train_conf['rank'] = int(os.getenv('NODE_RANK')) if os.getenv('NODE_RANK') is not None else 0
    train_conf['is_dist'] = train_conf['gpus'] > 1 or train_conf['num_nodes'] > 1

    train_conf['workers'] = min(available_cpu_count(), 16)

    # get training params
    train_conf['local_bs'] = train_conf['cumulative_bs'] // (train_conf['num_nodes'] * train_conf['gpus'])
    train_conf['lr'] = train_conf['base_lr'] * train_conf['cumulative_bs'] / 256

    run_name = f'LinCls-{opt.dataset}-{opt.run_name}'
    checkpoint_dir = f'{opt.project_path}evaluations/classification/runs/{run_name}/'

    # create or load model
    if opt.resume_from is not None:
        model = LinCls.load_from_checkpoint(checkpoint_path=opt.resume_from, pretrained_path=opt.encoder_path,
                                            train_conf=train_conf, dataset=opt.dataset)
    else:
        model = LinCls(pretrained_path=opt.encoder_path, train_conf=train_conf, dataset=opt.dataset)

    # init callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename='{epoch:02d}',
                                          monitor='validation/loss', mode='min', save_top_k=1, save_last=True,
                                          save_on_train_epoch_end=True)

    callbacks = [LearningRateMonitor(), checkpoint_callback]

    # logger
    # wandb logging
    offline = not opt.logging
    p_name = 'mi_ml_gen'
    if train_conf['rank'] == 0:
        if opt.wandb_id is not None:
            logger = WandbLogger(project=p_name, name=run_name, offline=offline, id=opt.wandb_id, resume='must')
        else:
            logger = WandbLogger(project=p_name, name=run_name, offline=offline)
    else:
        logger = WandbLogger(project=p_name, name=run_name, offline=True)

    # init trainer
    strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
    trainer = pl.Trainer(strategy=strategy, accelerator='gpu',
                         num_nodes=train_conf['num_nodes'], devices=train_conf['gpus'],
                         callbacks=callbacks, deterministic=True, logger=logger,
                         check_val_every_n_epoch=5, max_epochs=train_conf['max_epochs'])

    # data management
    train_file = f'{opt.data_path}/train.beton'
    val_file = f'{opt.data_path}/validation.beton'

    data_module = get_datamodule('bigbigan' if opt.dataset == 'imagenet' else None,
                                 train_conf['local_bs'], train_conf['workers'],
                                 opt.seed, train_conf['is_dist'], False, train_file, val_file)

    print(f"train_conf:\n{train_conf}")

    trainer.fit(model, data_module, ckpt_path=opt.resume_from)


if __name__ == '__main__':

    main(init_run())
