from skimage.transform import rescale
import torch

from torch.utils.data import Dataset
from functools import partial
import numpy as np

class UNetDataset(Dataset):
    def __init__(self, X, Y, transform=None, dist = None):
        self.transform = transform
        self._X = X
        self._Y = Y
        if dist is not None:
            self._dist = np.asarray(dist)

    def __getitem__(self, idx):
        if self._dist is not None:
            image = self._X[idx]
            mask = self._Y[idx]
            dist = self._dist[idx]
            sample = {'image': image, 'mask': mask, 'dist': dist}
        else:
            image = self._X[idx]
            mask = self._Y[idx]
            sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self._X)


class ChannelsFirst:
    def __call__(self, sample):
        if len(sample.keys())==3:
            image, mask, dist = sample['image'], sample['mask'], sample['dist']
        elif len(sample.keys()) ==2:
            image, mask = sample['image'], sample['mask']
        image = image.swapaxes(2,0)
        mask = mask.swapaxes(2,0)

        if len(sample.keys())==3:
            sample_out = {'image': image, 'mask': mask, 'dist': dist}
        elif len(sample.keys())==2:
            sample_out = {'image': image, 'mask': mask}
        return sample_out


class Rescale:

    def __init__(self, scale):
        assert isinstance(scale, float)
        self.output_scale = scale

    def __call__(self, sample):
        if len(sample.keys()) == 3:
            image, mask, dist = sample['image'], sample['mask'], sample['dist']
        elif len(sample.keys()) ==2:
            image, mask = sample['image'], sample['mask']

        resizer = partial(rescale, scale=self.output_scale, anti_aliasing=True, multichannel=True)
        out_image = resizer(image)
        out_mask = resizer(mask)

        if len(sample.keys()) == 3:
            sample_out = {'image': out_image, 'mask': out_mask, 'dist': dist}
        elif len(sample.keys()) == 2:
            sample_out = {'image': out_image, 'mask': out_mask}
        return sample_out



class ToTensor:

    def __call__(self, sample):
        if len(sample.keys()) == 3:
            image, mask, dist = sample['image'], sample['mask'], sample['dist']
        elif len(sample.keys()) == 2:
            image, mask = sample['image'], sample['mask']

        img_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        if dist:
            dist_tensor = torch.from_numpy(dist)

        if len(sample.keys()) == 3:
            sample_out = {'image': img_tensor, 'mask': mask_tensor, 'dist': dist_tensor}
        elif len(sample.keys()) == 2:
            sample_out = {'image': img_tensor, 'mask': mask_tensor}
        return sample_out