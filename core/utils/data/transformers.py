from skimage.transform import rescale

import torch
from functools import partial
import numpy as np

IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)


class ChannelsFirst:
    def __call__(self, sample):
        if len(sample.keys()) == 3:
            image, mask, dist = sample['image'], sample['mask'], sample['dist']
        elif len(sample.keys()) == 2:
            image, mask = sample['image'], sample['mask']

        if len(image.shape) == 3:
            image_sw1 = image.swapaxes(2, 0)
            image_sw2 = image_sw1.swapaxes(2,1)
            image_out = image_sw2
            mask_sw1 = mask.swapaxes(2, 0)
            mask_sw2 = mask_sw1.swapaxes(2,1)
            mask_out = mask_sw2

        elif len(image.shape) == 2:
            image_out = image[np.newaxis, ...]
            mask_out = mask[np.newaxis, ...]

        if len(sample.keys()) == 3:
            sample_out = {'image': image_out, 'mask': mask_out, 'dist': dist}
        elif len(sample.keys()) == 2:
            sample_out = {'image': image_out, 'mask': mask_out}
        return sample_out


class Rescale:

    def __init__(self, scale):
        assert isinstance(scale, float)
        self.output_scale = scale

    def __call__(self, sample):
        if len(sample.keys()) == 3:
            image, mask, dist = sample['image'], sample['mask'], sample['dist']
        elif len(sample.keys()) == 2:
            image, mask = sample['image'], sample['mask']

        if len(image.shape) == 3:
            resizer = partial(rescale, scale=self.output_scale, anti_aliasing=True, multichannel=True)
            out_image = resizer(image)
            out_mask = resizer(mask)
        elif len(image.shape) == 2:
            resizer = partial(rescale, scale=self.output_scale, anti_aliasing=True, multichannel=False)
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
        return sample_out






class Cut:

    def __init__(self, cut=True):
        assert isinstance(cut, bool)
        self.cut = cut

    def __call__(self, sample):
        if len(sample.keys()) == 3:
            image, mask, dist = sample['image'], sample['mask'], sample['dist']
        elif len(sample.keys()) == 2:
            image, mask = sample['image'], sample['mask']
        elif len(sample.keys()) ==1:
            image = sample['image']
            mask = None

        if self.cut:
            out_image = image[ROW_SLICE, COL_SLICE]
            if mask:
                out_mask = mask[ROW_SLICE, COL_SLICE]
        else:
            out_image = image
            if mask:
                out_mask = mask

        if len(sample.keys()) == 3:
            sample_out = {'image': out_image, 'mask': out_mask, 'dist': dist}
        elif len(sample.keys()) == 2:
            sample_out = {'image': out_image, 'mask': out_mask}
        elif len(sample.keys()) == 1:
            sample_out = {'image': out_image}
        return sample_out
