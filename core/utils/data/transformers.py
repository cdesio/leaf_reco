from skimage.transform import rescale

import torch
from functools import partial
import numpy as np

IMG_WIDTH = 1400
IMG_HEIGHT = 1400
#ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)
ROW_SLICE = slice(1000, 2400)

class ChannelsFirst:
    def __call__(self, sample):
        if 'image' in sample.keys():
            image = sample['image']

        if 'mask' in sample.keys():
            mask = sample['mask']
        else:
            mask = None
        if 'dist' in sample.keys():
            dist = sample['dist']
        else:
            dist = None

        if len(image.shape) == 3:
            image_sw1 = image.swapaxes(2, 0)
            image_sw2 = image_sw1.swapaxes(2,1)
            image_out = image_sw2
            if mask is not None:
                mask_sw1 = mask.swapaxes(2, 0)
                mask_sw2 = mask_sw1.swapaxes(2,1)
                mask_out = mask_sw2

        elif len(image.shape) == 2:
            image_out = image[np.newaxis, ...]
            if mask is not None:
                mask_out = mask[np.newaxis, ...]

        if 'mask' in sample.keys() and 'dist' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out, 'dist': dist}
        elif 'dist' not in sample.keys() and 'mask' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out}
        elif 'dist' not in sample.keys() and 'mask' not in sample.keys():
            sample_out = {'image': image_out}
        return sample_out


class Rescale:

    def __init__(self, scale):
        assert isinstance(scale, float)
        self.output_scale = scale

    def __call__(self, sample):
        if 'image' in sample.keys():
            image = sample['image']

        if 'mask' in sample.keys():
            mask = sample['mask']
        else:
            mask = None
        if 'dist' in sample.keys():
            dist = sample['dist']
        else:
            dist = None

        if len(image.shape) == 3:
            resizer = partial(rescale, scale=self.output_scale, anti_aliasing=True, multichannel=True)
            image_out = resizer(image)
            if mask is not None:
                mask_out = resizer(mask)
        elif len(image.shape) == 2:
            resizer = partial(rescale, scale=self.output_scale, anti_aliasing=True, multichannel=False)
            image_out = resizer(image)
            if mask is not None:
                mask_out = resizer(mask)

        if 'mask' in sample.keys() and 'dist' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out, 'dist': dist}
        elif 'dist' not in sample.keys() and 'mask' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out}
        elif 'dist' not in sample.keys() and 'mask' not in sample.keys():
            sample_out = {'image': image_out}
        return sample_out


class ToTensor:

    def __call__(self, sample):
        if len(sample.keys()) == 3:
            image, mask, dist = sample['image'], sample['mask'], sample['dist']
            img_tensor = torch.from_numpy(image)
            mask_tensor = torch.from_numpy(mask)
            # dist_tensor = torch.from_numpy(np.unique(dist, return_inverse=True)[1])
            dist_tensor = torch.from_numpy(np.asarray(dist))
            sample_out = {'image': img_tensor, 'mask': mask_tensor, 'dist': dist_tensor}
        elif len(sample.keys()) == 2:
            image, mask = sample['image'], sample['mask']
            img_tensor = torch.from_numpy(image)
            mask_tensor = torch.from_numpy(mask)
            sample_out = {'image': img_tensor, 'mask': mask_tensor}
        elif len(sample.keys()) == 1:
            image = sample['image']
            img_tensor = torch.from_numpy(image)
            sample_out = {'image': img_tensor}
        return sample_out






class Cut:

    def __init__(self, cut=True, row_slice=ROW_SLICE, col_slice=COL_SLICE):
        assert isinstance(cut, bool)
        self.cut = cut
        self.row_slice = row_slice
        self.col_slice = col_slice

    def __call__(self, sample):
        if 'image' in sample.keys():
            image = sample['image']

        if 'mask' in sample.keys():
            mask = sample['mask']
        else:
            mask=None
        if 'dist' in sample.keys():
            dist = sample['dist']
        else:
            dist=None

        if self.cut:
            image_out = image[self.row_slice, self.col_slice]
            if mask is not None:
                mask_out = mask[self.row_slice, self.col_slice]
        else:
            image_out = image
            if mask is not None:
                mask_out = mask

        if 'mask' in sample.keys() and 'dist' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out, 'dist': dist}
        elif 'dist' not in sample.keys() and 'mask' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out}
        elif 'dist' not in sample.keys() and 'mask' not in sample.keys():
            sample_out = {'image': image_out}
        return sample_out
