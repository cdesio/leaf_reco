from skimage.transform import rescale
import torch

from torch.utils.data import Dataset
from functools import partial


class UNetDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.transform = transform
        self._X = X
        self._Y = Y

    def __getitem__(self, idx):
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
        image, mask = sample['image'], sample['mask']

        image = image.swapaxes(2,0)
        mask = mask.swapaxes(2,0)

        return {'image': image,
                'mask': mask}


class Rescale:

    def __init__(self, scale):
        assert isinstance(scale, float)
        self.output_scale = scale

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        resizer = partial(rescale, scale=self.output_scale, anti_aliasing=True, multichannel=True)
        out_image = resizer(image)
        out_mask = resizer(mask)

        return {'image': out_image,
                'mask': out_mask}


class ToTensor:

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        img_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return {'image': img_tensor,
                'mask': mask_tensor}