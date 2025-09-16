from typing import Any

import numpy as np
from kornia.augmentation import AugmentationSequential, RandomResizedCrop, RandomHorizontalFlip, Normalize, \
    CenterCrop
import pytorch_lightning as pl
import torchvision.models as models
import torch
from scheduling_utils.schedulers import CosineScheduler
from torch import nn

n_classes = {
    'birdsnap': 525,
    'caltech101': 101,
    'cifar100': 100,
    'DTD': 47,
    'fgvc-aircraft-2013b': 70,
    'StanfordCars': 196,
    'flowers102': 102,
    'food101': 101,
    'imagenet': 1000,
    'imagenet100': 100,
    'pets': 37
}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LinCls(pl.LightningModule):
    """
    Build a Linear Classifier over a pretrained feature extractor.
    # Copyright (c) Facebook, Inc. and its affiliates.
    # All rights reserved.

    # This source code is licensed under the license found in the
    # LICENSE file in the root directory of this source tree.

    Original Code ported to Pytorch-Lightning
    """

    def __init__(self, pretrained_path: str | None, train_conf: dict, dataset: str, image_resolution: int = 128):

        super().__init__()

        # create model
        model = models.resnet50()

        # init the fc layer
        model.fc = nn.Linear(in_features=2048, out_features=n_classes[dataset])
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        # load encoder weights from pre-trained (for testing you can pass none, since weights will be loaded later)
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location="cuda")

            # rename pre-trained keys
            state_dict = checkpoint['state_dict']

            is_byol = False

            for k in list(state_dict.keys()):

                # retain only encoder up to before the embedding layer
                if k.startswith('encoder') and not k.startswith('encoder.fc'):
                    # remove prefix
                    state_dict[k[len("encoder."):]] = state_dict[k]
                elif k.startswith('model') and not k.startswith('model.fc'):
                    # remove prefix
                    state_dict[k[len("model."):]] = state_dict[k]
                elif k.startswith('online_network.encoder.'):
                    # BYOL case
                    is_byol = True

                    # remove prefix
                    state_dict[k[len("online_network.encoder."):]] = state_dict[k]

                # delete renamed or unused k
                del state_dict[k]

            # convert custom Resnet50 weights back to standard Resnet50
            # TODO this is because the code used for BYOL changes the param names of Resnet50
            #      This whole operation can be avoided by modification of the byol model code.
            if is_byol:

                original_state_dict = model.state_dict()
                custom_state_dict = state_dict

                # Step 4: Map custom model's state dictionary to original ResNet-50 names
                mapped_state_dict = {}
                original_keys = list(original_state_dict.keys())
                custom_keys = list(custom_state_dict.keys())

                for i, custom_key in enumerate(custom_keys):
                    original_key = original_keys[i]
                    mapped_state_dict[original_key] = custom_state_dict[custom_key]

                state_dict = mapped_state_dict

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        self.model = model

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # LR and Scheduler options
        self.lr = train_conf["lr"]
        self.final_lr = self.lr / 10
        self.decay = train_conf["decay"]
        self.decay_epochs = train_conf["max_epochs"]
        self.scheduler = None

        # Augmentations
        crop_size = int(image_resolution * 0.875)
        self.train_augmentations = AugmentationSequential(
            RandomResizedCrop((crop_size, crop_size), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            same_on_batch=False)

        self.val_augmentations = AugmentationSequential(
            CenterCrop((crop_size, crop_size)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    def forward(self, images, target):

        # Switch to eval mode:
        # Under the protocol of linear classification on frozen features/models,
        # it is not legitimate to change any part of the pre-trained model.
        # BatchNorm in train mode may revise running mean/std (even if it receives
        # no gradient), which are part of the model parameters too.
        self.model.eval()

        output = self.model(images)
        loss = self.criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        return loss, acc1, acc5

    def training_step(self, batch, batch_idx):

        images, labels = self.preprocess_batch(batch, mode='train')
        b = labels.shape[0]

        loss, acc1, acc5 = self.forward(images, labels)

        self.log('train/loss', loss, sync_dist=True, on_step=False, on_epoch=True, batch_size=b)
        self.log('train/acc1', acc1, sync_dist=True, on_step=False, on_epoch=True, batch_size=b)
        self.log('train/acc5', acc5, sync_dist=True, on_step=False, on_epoch=True, batch_size=b)

        return loss

    def validation_step(self, batch, batch_idx):

        images, labels = self.preprocess_batch(batch, mode='val')
        b = labels.shape[0]

        loss, acc1, acc5 = self.forward(images, labels)

        self.log('validation/loss', loss, sync_dist=True, on_step=False, on_epoch=True, batch_size=b)
        self.log('validation/acc1', acc1, sync_dist=True, on_step=False, on_epoch=True, batch_size=b)
        self.log('validation/acc5', acc5, sync_dist=True, on_step=False, on_epoch=True, batch_size=b)

        return loss

    def on_test_start(self) -> None:

        # create list of results
        self.test_preds = np.empty((0, self.model.fc.out_features))
        self.test_labels = np.empty((0, ))

    def test_step(self, batch, batch_idx):

        images, labels = self.preprocess_batch(batch, mode='val')

        preds = self.model(images)

        self.test_preds = np.concatenate([self.test_preds, preds.cpu().numpy()], axis=0)
        self.test_labels = np.concatenate([self.test_labels, labels.cpu().numpy()], axis=0)

    def on_test_epoch_end(self):

        acc1, acc5 = accuracy(torch.from_numpy(self.test_preds), torch.from_numpy(self.test_labels), topk=(1, 5))
        self.log('Top1', float(f'{acc1.item():.2f}'))
        self.log('Top5', float(f'{acc5.item():.2f}'))

    def configure_optimizers(self):

        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias

        optimizer = torch.optim.SGD(parameters, self.lr, momentum=0.9, weight_decay=0.)
        return {'optimizer': optimizer}

    def on_train_start(self):

        # init warmup/decay lr
        decay_end = self.decay_epochs * self.trainer.num_training_batches

        if self.decay:

            decay_start = 0
            self.scheduler = CosineScheduler(decay_start, decay_end, self.lr, self.final_lr)

    def on_train_batch_start(self, _: Any, batch_idx: int):

        # adjust LR
        current_step = (self.current_epoch * self.trainer.num_training_batches) + batch_idx

        for param_group in self.optimizers().optimizer.param_groups:
            if self.scheduler is None:
                param_group['lr'] = self.lr
            else:
                param_group['lr'] = self.scheduler.step(current_step)

    def preprocess_batch(self, batch, mode):

        images, labels = batch

        if mode == 'train':
            images = self.train_augmentations(images)
        else:
            assert mode == 'val'
            images = self.val_augmentations(images)

        return images, labels.squeeze(1)
