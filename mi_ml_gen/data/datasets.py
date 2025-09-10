import pathlib

from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset

from PIL import Image


class FakeDataset(Dataset):

    def __init__(self, n_samples: int, ):
        """
        Fake Dataset that does nothing.
        Used to combine online learning with PL (which requires DataLoader).
        """
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (idx, )


class ImageDataset(Dataset):

    def __init__(self, folder: str, image_size: int):
        """
        :param folder: path to images
        :param image_size: for resizing
        """

        self.samples = sorted(list(pathlib.Path(folder).rglob('*.png')) + list(pathlib.Path(folder).rglob('*.jpg')) +
                              list(pathlib.Path(folder).rglob('*.bmp')) + list(pathlib.Path(folder).rglob('*.JPEG')))

        self.transforms = Compose([ToTensor(), Resize(image_size, antialias=True)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # path to string
        image_path = self.samples[idx].absolute().as_posix()

        try:
            image = Image.open(image_path).convert('RGB')
            return (image,)
        except OSError as e:
            print(f'OS ERROR: {e}. Skipping file {image_path}')


class ImageLabelDataset(Dataset):

    def __init__(self, folder: str, image_size: int):
        """
        :param folder: path to images
        :param image_size: for resizing
        """

        self.samples = sorted(list(pathlib.Path(folder).rglob('*.png')) + list(pathlib.Path(folder).rglob('*.jpg')) +
                              list(pathlib.Path(folder).rglob('*.bmp')) + list(pathlib.Path(folder).rglob('*.JPEG')))

        self.transforms = Compose([ToTensor(), Resize(image_size, antialias=True)])
        self.class_mapping = sorted(list(set([image_path.absolute().as_posix().split('/')[-2]
                                              for image_path in self.samples])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # path to string
        image_path = self.samples[idx].absolute().as_posix()

        image = Image.open(image_path).convert('RGB')

        class_name = image_path.split('/')[-2]
        class_index = self.class_mapping.index(class_name)

        return (image, class_index)
