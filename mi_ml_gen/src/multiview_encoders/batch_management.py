import numpy as np
import torch
from einops import pack

from kornia.augmentation import AugmentationSequential, CenterCrop, Normalize, RandomHorizontalFlip, \
    ColorJiggle, RandomGrayscale, RandomResizedCrop

from src.generators.sampling import generate_anchors, get_device_specific_seeds
from src.generators.utils import load, disable_train, decode, to_device, z_to_w
from src.noise_maker.walkers import SynthWalker, RandomWalker, MultipleNonLinearWalker, MultipleRandomWalker

image_resolution = 128


class BatchManager:

    def __init__(self, data_conf: dict):
        """
        taking a data conf, allows to generate anchors and positives according to conf.
        """

        # generation procedure Tz
        if data_conf['generator_path'] is None:

            # real data case
            self.generator = None
            self.anchor_params = None
            self.Tz = lambda z, **kwargs: (None, z, None)

        else:

            # load bigbigan or stylegan generator
            self.generator = load(data_conf["data_type"], data_conf["generator_path"])

            # tuple object
            self.anchor_params = data_conf["anchor_params"]

            if data_conf["Tz"] is None:
                # baseline without Tz = synth data only
                self.Tz = SynthWalker(self.generator.type)

            elif len(data_conf["Tz"].keys()) == 1 and isinstance(list(data_conf["Tz"].values())[0], str):

                # baseline "learned" case, only one Tz which is a path to a checkpoint
                self.Tz = torch.load(list(data_conf["Tz"].values())[0])['nn_walker']

                # freeze all layers
                for name, param in self.Tz.named_parameters():
                    param.requires_grad = False

                self.Tz.walker.training = False
                self.Tz.train = disable_train  # prevent PL to set training true

            elif len(data_conf["Tz"].keys()) == 1 and isinstance(list(data_conf["Tz"].values())[0], list):

                # baseline "random" case, only one Tz which is a list with gauss params
                self.Tz = RandomWalker(data_conf["data_type"], 
                                       list(data_conf["Tz"].keys())[0], 
                                       list(data_conf["Tz"].values())[0])
                self.Tz.training = False
                self.Tz.train = disable_train  # prevent PL to set training true

            elif len(data_conf["Tz"].keys()) > 1 and isinstance(list(data_conf["Tz"].values())[0], str):

                # "learned" case with different checkpoints on each group
                # Data conf is a dict with groups: chunks -> ckpt path
                self.Tz = MultipleNonLinearWalker(data_conf["Tz"])
                self.Tz.training = False
                self.Tz.train = disable_train  # prevent PL to set training true

            elif len(data_conf["Tz"].keys()) > 1 and isinstance(list(data_conf["Tz"].values())[0], list):

                # "random" case with different perturbations on each chunk/group
                # Data conf is a dict with groups: chunks -> tuple
                self.Tz = MultipleRandomWalker(data_conf["data_type"], data_conf["Tz"])
                self.Tz.training = False
                self.Tz.train = disable_train  # prevent PL to set training true
            else:
                raise NotImplementedError(f'unknown Tz conf: {data_conf["Tz"]}')

        # data space augmentations Tx
        crop_size = int(image_resolution * 0.875)

        if data_conf["Tx"] is None:
            self.Tx = AugmentationSequential(
                CenterCrop((crop_size, crop_size)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                same_on_batch=False)

        elif data_conf["Tx"] == "no_crop":
            self.Tx = AugmentationSequential(
                CenterCrop((crop_size, crop_size)),
                RandomHorizontalFlip(),
                ColorJiggle(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                same_on_batch=False)

        elif data_conf["Tx"] == "no_color":
            self.Tx = AugmentationSequential(
                RandomResizedCrop((crop_size, crop_size), scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                same_on_batch=False)

        elif data_conf["Tx"] == "simclr":
            self.Tx = AugmentationSequential(
                RandomResizedCrop((crop_size, crop_size), scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                ColorJiggle(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                same_on_batch=False)
        else:
            raise AttributeError(f"Tx configuration '{data_conf['Tx']}' is not recognized")

    def to_device(self, device: int):

        # ensure batch is generated directly on correct device
        self.generator = to_device(self.generator, device) if self.generator is not None else None

        self.Tz = self.Tz.to(device) if self.generator is not None else self.Tz
        if isinstance(self.Tz, MultipleNonLinearWalker):
            for i in range(len(self.Tz.all_walkers)):
                if self.Tz.all_walkers[i] is not None:
                    self.Tz.all_walkers[i].to(device)
        self.Tx = self.Tx.to(device)

    @torch.no_grad()
    def generate_batch(self, batch: torch.Tensor, rank: int, epoch: int):

        # if batch has real data is tensor of shape (b, c, h, w), else is tensor of shape (b, 1)
        if len(batch.shape) == 4:
            # no need to generate anchors - apply Tz
            return self.Tx(batch), self.Tx(batch)

        # we need to generate batch from GAN
        b, _ = batch.shape

        seeds = get_device_specific_seeds(rank, epoch)

        z_anchors, _ = generate_anchors(self.generator, b, self.anchor_params, seeds['anchor'], only_z=True)

        # now get positives.
        if self.generator.type == 'bigbigan':

            _, z_positives, _ = self.Tz(z_anchors, seed=seeds['positive'])

            if isinstance(z_anchors, np.ndarray):
                z_anchors = torch.from_numpy(z_anchors).to(batch.device)

            if isinstance(z_positives, np.ndarray):
                z_positives = torch.from_numpy(z_positives).to(batch.device)

            all_z, _ = pack([z_anchors, z_positives], '* d')
            all_images = decode(self.generator, all_z)

        else:

            w_anchors = z_to_w(self.generator.mapping, z_anchors)
            _, w_positives, _ = self.Tz(z_anchors, seed=seeds['positive'], mapping_net=self.generator.mapping)

            all_images = decode(self.generator, pack([w_anchors, w_positives], '* n d')[0])

        img_anchors = all_images[:b]
        img_positives = all_images[b:]

        # apply Tx and return
        return self.Tx(img_anchors.to(batch.device)), self.Tx(img_positives.to(batch.device))
