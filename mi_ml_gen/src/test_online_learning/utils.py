import os
import warnings

from kornia.augmentation import AugmentationSequential, RandomResizedCrop, RandomHorizontalFlip, ColorJiggle, \
    RandomGrayscale, Normalize

from torch import distributed as dist


def setup(rank, world_size):

    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        if rank == 0:
            warnings.warn("Set Environ Variable 'MASTER_ADDR'='localhost'")

    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
        if rank == 0:
            warnings.warn("Set Environ Variable 'MASTER_PORT'='29500'")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


class Tx:
    """
    kornia transforms
    """
    def __init__(self):

        crop_size = 112
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.img_transform = AugmentationSequential(
            RandomResizedCrop((crop_size, crop_size), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJiggle(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2),
            Normalize(mean=mean, std=std),
            same_on_batch=False)

    def apply(self, imgs):
        return self.img_transform(imgs)
