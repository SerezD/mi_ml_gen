import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import make_grid
import pytorch_lightning as pl

from scheduling_utils.schedulers import CosineScheduler, LinearCosineScheduler

from typing import Any
import wandb

from src.multiview_encoders.batch_management import BatchManager


class SimSiam(pl.LightningModule):
    """
    Build a SimSiam model.
    
    Copyright (c) Facebook, Inc. and its affiliates.
    All rights reserved.

    This source code is licensed under the license found in the
    LICENSE file in the root directory of this source tree.

    Original Code ported to Pytorch-Lightning from https://github.com/facebookresearch/simsiam
    """

    def __init__(self, model_conf: dict, train_conf: dict, data_conf: dict, base_encoder=models.resnet50):

        super().__init__()

        dim = int(model_conf["dim"])
        pred_dim = int(model_conf["pred"])

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

        # define loss function (criterion)
        self.criterion = nn.CosineSimilarity(dim=1)

        # LR and Scheduler options
        self.lr = train_conf["lr"]
        self.final_lr = self.lr / 10
        self.decay, self.warmup = train_conf["decay"], train_conf["warmup"]
        self.warmup_epochs = train_conf["max_epochs"] // 10
        self.decay_epochs = train_conf["max_epochs"]
        self.scheduler = None

        # batch_generator
        self.batch_generator = BatchManager(data_conf)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()

    def on_train_epoch_start(self) -> None:

        device = self.trainer.local_rank
        self.batch_generator.to_device(device)

    def training_step(self, batch, batch_idx):

        views_1, views_2 = self.batch_generator.generate_batch(batch[0], self.global_rank, self.current_epoch)

        if batch_idx == 0 and self.current_epoch % 5 == 0:

            display = torch.cat([views_1[:8], views_2[:8]], dim=0)
            display = make_grid(display)
            display = wandb.Image(display)

            self.logger.experiment.log({'views': display})

        # compute output and loss
        p1, p2, z1, z2 = self.forward(x1=views_1, x2=views_2)
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

        self.log('train/loss', loss, sync_dist=True, on_epoch=True, on_step=False)

        return loss

    # Define Optimizer and Scheduler Stuff
    def configure_optimizers(self):

        optim_params = [{'params': self.encoder.parameters(), 'fix_lr': False},
                        {'params': self.predictor.parameters(), 'fix_lr': True}]

        optimizer = torch.optim.SGD(optim_params, self.lr, momentum=0.9, weight_decay=1e-4)

        return optimizer

    def on_train_start(self):

        # init warmup/decay lr
        decay_end = self.decay_epochs * self.trainer.num_training_batches

        if self.warmup and self.decay:

            warmup_start = 0
            warmup_end = self.warmup_epochs * self.trainer.num_training_batches

            self.scheduler = LinearCosineScheduler(warmup_start, decay_end, self.lr, self.final_lr, warmup_end)

        elif self.decay:

            decay_start = 0
            self.scheduler = CosineScheduler(decay_start, decay_end, self.lr, self.final_lr)

    def on_train_batch_start(self, _: Any, batch_idx: int):

        # adjust LR
        current_step = (self.current_epoch * self.trainer.num_training_batches) + batch_idx

        for param_group in self.optimizers().optimizer.param_groups:
            if self.scheduler is None or ('fix_lr' in param_group and param_group['fix_lr']):
                param_group['lr'] = self.lr
            else:
                param_group['lr'] = self.scheduler.step(current_step)
