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
from src.multiview_encoders.byol.model import Byol
from src.multiview_encoders.simsiam.model import SimSiam
from src.multiview_encoders.utils.system import available_cpu_count, set_matmul_precision


def init_run():

    parser = argparse.ArgumentParser(description="script for multiview representation learning Encoders training.")

    parser.add_argument('--data_path', type=str, 
                        help="path to a folder containing the 'fake_train' and 'real_train' beton files.")

    parser.add_argument('--seed', type=int, help='seed for randomness reproducibility')

    parser.add_argument('--conf', type=str, help='name of the model conf file - without extension')

    parser.add_argument('--encoder', type=str, choices=['simsiam', 'byol'], help='encoder type')

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
    return conf['data'], conf['train'], conf['model']


def main(opt):

    pl.seed_everything(opt.seed, workers=True)

    set_matmul_precision()

    opt.project_path = str(pathlib.Path(__file__).parent.resolve()).split('src')[0]

    # get model config
    conf_path = f'{opt.project_path}configurations/encoders/{opt.conf}.yaml'
    data_conf, train_conf, model_conf = get_model_config(conf_path)

    # config variables
    train_conf['num_nodes'] = int(os.getenv('NODES')) if os.getenv('NODES') is not None else 1
    train_conf['gpus'] = torch.cuda.device_count()
    train_conf['rank'] = int(os.getenv('NODE_RANK')) if os.getenv('NODE_RANK') is not None else 0
    train_conf['is_dist'] = train_conf['gpus'] > 1 or train_conf['num_nodes'] > 1

    train_conf['workers'] = min(available_cpu_count(), 16)

    # get training params
    train_conf['local_bs'] = train_conf['cumulative_bs'] // (train_conf['num_nodes'] * train_conf['gpus'])
    train_conf['lr'] = train_conf['base_lr'] * train_conf['cumulative_bs'] / 256

    # create or load model
    if opt.resume_from is not None:
        if opt.encoder == 'simsiam':
            model = SimSiam.load_from_checkpoint(checkpoint_path=opt.resume_from, model_conf=model_conf,
                                                 train_conf=train_conf, data_conf=data_conf)
        else:
            model = Byol.load_from_checkpoint(checkpoint_path=opt.resume_from, model_conf=model_conf,
                                              train_conf=train_conf, data_conf=data_conf)
    else:
        # init model
        if opt.encoder == 'simsiam':
            model = SimSiam(model_conf, train_conf, data_conf)
        else:
            model = Byol(model_conf, train_conf, data_conf)

    # init callbacks
    save_dir = f'{opt.project_path}multiview_encoders/runs/{opt.conf}/'
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename='{epoch:02d}', save_last=True,
                                          every_n_epochs=5, save_on_train_epoch_end=True)

    callbacks = [LearningRateMonitor(), checkpoint_callback]

    # wandb logging
    offline = not opt.logging
    p_name = 'mi_ml_gen'
    if train_conf['rank'] == 0:
        if opt.wandb_id is not None:
            logger = WandbLogger(project=p_name, name=opt.conf, offline=offline, id=opt.wandb_id, resume='must')
        else:
            logger = WandbLogger(project=p_name, name=opt.conf, offline=offline)
    else:
        logger = WandbLogger(project=p_name, name=opt.conf, offline=True)

    # init trainer
    strategy = DDPStrategy(find_unused_parameters=False, static_graph=True, gradient_as_bucket_view=True)

    trainer = pl.Trainer(strategy=strategy, 
                         accelerator='gpu',
                         num_nodes=train_conf['num_nodes'], 
                         devices=train_conf['gpus'],
                         callbacks=callbacks, 
                         deterministic=True, 
                         logger=logger,
                         sync_batchnorm=train_conf['is_dist'],
                         check_val_every_n_epoch=5, 
                         max_epochs=train_conf['max_epochs'])

    # data management
    is_fake = data_conf["generator_path"] is not None  # if None means need real data
    train_file = f'{opt.data_path}/fake_train.beton' if is_fake else f'{opt.data_path}/train.beton'

    data_module = get_datamodule(data_conf["data_type"], train_conf['local_bs'], train_conf['workers'],
                                 opt.seed, train_conf['is_dist'], is_fake, train_file)

    print(f"train_conf:\n{train_conf}")
    print(f"data_conf:\n{data_conf}")

    trainer.fit(model, data_module, ckpt_path=opt.resume_from)


if __name__ == '__main__':

    main(init_run())
