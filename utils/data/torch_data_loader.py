import torch

import numpy as np
from torch.utils.data import Dataset


class UNet_Dataset(Dataset):
    def __init__(self, transform=None):
        self.infile = infile

        self.transform = transform
        # def __len__(self):
        #    return len(self.input_images)

    def __getitem__(self, idx):
        data = np.load(self.infile)
        image = data["x"][idx]
        mask = data["y"][idx]
        sample = {'image': image, 'masks': mask}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(np.load(self.infile))


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['masks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'masks': torch.from_numpy(mask)}


dataset = UNet_Dataset(infile="Xy_train_clean_300_24_10_25.npz", transform=ToTensor())
