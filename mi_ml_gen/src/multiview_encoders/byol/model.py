import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.utils import make_grid
import pytorch_lightning as pl

from scheduling_utils.schedulers import CosineScheduler, LinearCosineScheduler

from typing import Any
import wandb

from src.multiview_encoders.batch_management import BatchManager


class MLPHead(nn.Module):

    def __init__(self, in_channels, mlp_hidden_size, projection_size):

        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size, bias=False),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class ResNet50(torch.nn.Module):

    def __init__(self, projection_head):

        super().__init__()
        resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(resnet.fc.in_features, resnet.fc.in_features * 4, projection_head)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)


class Byol(pl.LightningModule):
    """
    Build a Byol model.

    Original Code ported to Pytorch-Lightning from https://github.com/sthalles/PyTorch-BYOL
    """

    def __init__(self, model_conf: dict, train_conf: dict, data_conf: dict):
        """

        """

        super().__init__()

        dim = int(model_conf["dim"])
        pred_dim = int(model_conf["pred"])

        self.online_network = ResNet50(projection_head=dim)
        self.target_network = ResNet50(projection_head=dim)
        self.predictor = MLPHead(in_channels=self.online_network.projection.net[-1].out_features,
                                 mlp_hidden_size=pred_dim, projection_size=dim)

        # target network
        self.momentum = 0.996  # momentum update

        # define loss function (criterion)
        self.criterion = lambda x, y: 2 - 2 * (F.normalize(x, dim=1) * F.normalize(y, dim=1)).sum(dim=-1)

        # LR and Scheduler options
        self.lr = train_conf["lr"]
        self.final_lr = self.lr / 10
        self.decay, self.warmup = train_conf["decay"], train_conf["warmup"]
        self.warmup_epochs = train_conf["max_epochs"] // 10
        self.decay_epochs = train_conf["max_epochs"]
        self.scheduler = None

        # batch_generator
        self.batch_generator = BatchManager(data_conf)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            loss
        """

        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(x1))
        predictions_from_view_2 = self.predictor(self.online_network(x2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(x1)
            targets_to_view_1 = self.target_network(x2)

        loss = self.criterion(predictions_from_view_1, targets_to_view_1)
        loss += self.criterion(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def training_step(self, batch, batch_idx):

        views_1, views_2 = self.batch_generator.generate_batch(batch[0], self.global_rank, self.current_epoch)

        if batch_idx == 0 and self.current_epoch % 5 == 0:

            display = torch.cat([views_1[:8], views_2[:8]], dim=0)
            display = make_grid(display)
            display = wandb.Image(display)

            self.logger.experiment.log({'views': display})

        # compute loss
        loss = self.forward(x1=views_1, x2=views_2)

        self.log('train/loss', loss, sync_dist=True, on_epoch=True, on_step=False)

        return loss

    # Define Optimizer and Scheduler Stuff
    def configure_optimizers(self):

        optim_params = list(self.online_network.parameters()) + list(self.predictor.parameters())

        optimizer = torch.optim.SGD(optim_params, self.lr, momentum=0.9, weight_decay=1e-4)

        return optimizer

    def on_train_start(self):

        # Initialize Target Network
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # init warmup/decay lr
        decay_end = self.decay_epochs * self.trainer.num_training_batches

        if self.warmup and self.decay:

            warmup_start = 0
            warmup_end = self.warmup_epochs * self.trainer.num_training_batches

            self.scheduler = LinearCosineScheduler(warmup_start, decay_end, self.lr, self.final_lr, warmup_end)

        elif self.decay:

            decay_start = 0
            self.scheduler = CosineScheduler(decay_start, decay_end, self.lr, self.final_lr)

    def on_train_epoch_start(self) -> None:

        device = self.trainer.local_rank
        self.batch_generator.to_device(device)

    def on_train_batch_start(self, _: Any, batch_idx: int):

        # adjust LR
        current_step = (self.current_epoch * self.trainer.num_training_batches) + batch_idx

        for param_group in self.optimizers().optimizer.param_groups:
            if self.scheduler is None or ('fix_lr' in param_group and param_group['fix_lr']):
                param_group['lr'] = self.lr
            else:
                param_group['lr'] = self.scheduler.step(current_step)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:

        # update the key encoder
        self._update_target_network_parameters()
