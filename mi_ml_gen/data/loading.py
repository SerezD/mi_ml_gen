from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.loader import OrderOption
from ffcv.reader import Reader
from ffcv.transforms import ToTensor, ToTorchImage
from ffcv_pl.ffcv_utils.augmentations import DivideImage255
from ffcv_pl.ffcv_utils.utils import FFCVPipelineManager
from kornia.augmentation import Resize, CenterCrop
from ffcv_pl.data_loading import FFCVDataModule
import torch

"""
IMPORTANT: Ensure the same preprocessing for loaded (real) and generated images! 

For Stylegan, first center crop and the resize. 

For Imagenet, BigBiGan output is directly 128. The "real" data is simply resized from 256x256
"""

stylegan_stored_size = 512
stylegan_crop = (384, 384)
bigbigan_crop = (256, 256)
image_resize = (128, 128)

stylegan_transforms = [CenterCropRGBImageDecoder(stylegan_crop, ratio=stylegan_crop[0] / stylegan_stored_size),
                       ToTensor(),
                       ToTorchImage(),
                       DivideImage255(dtype=torch.float32),
                       Resize(image_resize)]

bigbigan_transforms = [CenterCropRGBImageDecoder(bigbigan_crop, ratio=1),
                       ToTensor(),
                       ToTorchImage(),
                       DivideImage255(dtype=torch.float32),
                       Resize(image_resize)]

# preprocessing for classification.
classification_transforms = [CenterCropRGBImageDecoder(image_resize, ratio=1),
                             ToTensor(),
                             ToTorchImage(),
                             DivideImage255(dtype=torch.float32)]


def preprocess_stylegan_batch(images: torch.Tensor):

    # images are 512 x 512, Height is padded 512 x 384
    images = CenterCrop(stylegan_crop)(images)
    images = Resize(image_resize)(images)
    return images


def preprocess_bigbigan_batch(images: torch.Tensor):
    images = Resize(image_resize)(images)
    return images


def has_labels(filepath: str):
    # check if the passed beton file has labels or is image only
    n_fields = len(Reader(filepath).field_names)

    if n_fields > 2:
        raise RuntimeError(f'filepath: {filepath} points to a dataset with more than 2 returned items.')

    return n_fields == 2


def prepare_manager(file_path: str, data_type: str, is_fake: bool, ordering: OrderOption):
    """
    create and return FFCVPipelineManager Object
    """

    # no manager to pass
    if file_path is None:
        return None

    # prepare pipeline
    if has_labels(file_path):
        pipeline = [classification_transforms.copy() if data_type is None else bigbigan_transforms.copy(),
                    [IntDecoder(), ToTensor()]]

    else:
        assert data_type is not None, "data_type can't be None when training the Encoder!"

        if is_fake:
            pipeline = [[IntDecoder(), ToTensor()]]
        else:
            pipeline = [bigbigan_transforms.copy()] if data_type == 'bigbigan' else [stylegan_transforms.copy()]

    return FFCVPipelineManager(file_path, pipeline, ordering)


def get_datamodule(data_type: str | None, batch_size: int, workers: int, seed: int, is_dist: bool, is_fake: bool,
                   train_file: str = None, val_file: str = None, test_file: str = None, predict_file: str = None,
                   ):
    """
    :param data_type: 'stylegan' or 'bigbigan', None for classification/transfer learning
    :param batch_size: batch size of single loader
    :param workers: number of workers
    :param seed: for reproduction
    :param is_dist: pass True if using more than one gpu.
    :param is_fake: if True, loads fake dataset for train.
    :param train_file: path to .beton file, ignore Train Loader if None
    :param val_file: path to .beton file, ignore Val Loader if None
    :param test_file: path to .beton file, ignore Test Loader if None
    :param predict_file: path to .beton file, ignore Predict Loader if None
    """

    assert data_type is None or (data_type == 'bigbigan' or data_type == 'stylegan')

    return FFCVDataModule(batch_size, workers, is_dist,
                          train_manager=prepare_manager(train_file, data_type, is_fake, OrderOption.RANDOM),
                          val_manager=prepare_manager(val_file, data_type, False, OrderOption.SEQUENTIAL),
                          test_manager=prepare_manager(test_file, data_type, False, OrderOption.SEQUENTIAL),
                          predict_manager=prepare_manager(predict_file, data_type, False, OrderOption.SEQUENTIAL),
                          seed=seed)
