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

    def __init__(self, cut=True, row_slice=ROW_SLICE, col_slice=COL_SLICE, swap=False, flip_lr=False, flip_ud=False):
        assert isinstance(cut, bool)
        self.cut = cut
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.swap = swap
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

        if self.swap and self.flip_lr:
            image_out = np.fliplr(image_out.swapaxes(1,0))
            if mask is not None:
                mask_out = np.fliplr(mask_out.swapaxes(1,0))
        if self.swap and self.flip_ud:
            image_out = np.flipud(image_out.swapaxes(1,0))
            if mask is not None:
                mask_out = np.flipud(mask_out.swapaxes(1,0))

        if self.swap and not (self.flip_lr and self.flip_ud):
            image_out = image_out.swapaxes(1,0)
            if mask is not None:
                mask_out = mask_out.swapaxes(1,0)

        if self.flip_lr and not(self.swap and self.flip_ud):
            image_out = np.fliplr(image_out)
            if mask is not None:
                mask_out = np.fliplr(mask_out)
        if self.flip_ud and not(self.swap and self.flip_lr):
            image_out = np.flipud(image_out)
            if mask is not None:
                mask_out = np.flipud(mask_out)

         if self.flip_lr and self.flip_ud and not self.swap:
            image_out = np.flipud(np.fliplr(image_out))
            if mask in not None:
                mask_out = np.flipud(np.fliplr(mask_out))

        if self.flip_lr and self.flip_ud and self.swap:
            image_out = np.flipud(np.fliplr(image_out.swapaxes(0,1)))
            if mask in not None:
                mask_out = np.flipud(np.fliplr(mask_out.swapaxes(0.1)))

        if 'mask' in sample.keys() and 'dist' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out, 'dist': dist}
        elif 'dist' not in sample.keys() and 'mask' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out}
        elif 'dist' not in sample.keys() and 'mask' not in sample.keys():
            sample_out = {'image': image_out}
        return sample_out


class GaussianNoise:
    def __init__(self, add_noise=True, variance = 0.1):
        assert isinstance(add_noise, bool)
        self.add_noise = add_noise
        self.var = variance
    """
    def __init__(self, range):
        assert isinstance(range, tuple)
        self.corr_min, self.corr_max=range
        self.corrections = np.arange(self.corr_min, self.corr_max, 0.02, dtype=np.float16)
    def __call__(self, sample):
        if 'image' in sample.keys():
            image_corr = sample['image']
            corr = np.random.choice(self.corrections)
            image_corr =+ corr
            sample['image']= image_corr
        return sample

    """
    @staticmethod

    def noisy(image, var):
        import numpy as np
        """
         Gaussian-distributed additive noise.

        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        

        """
        if len(image.shape)==2:
           row, col = image.shape
        elif len(image.shape)==3:
            shape = image.shape
            if shape[-1]==1:
                row, col, _ = image.shape
            elif shape[0]==1:
                _, row, col = image.shape
        mean = 0
        #var = var
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy

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

        if self.add_noise:
            image_out = self.noisy(image, self.var)
            if mask is not None:
                mask_out = mask

        if 'mask' in sample.keys() and 'dist' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out, 'dist': dist}
        elif 'dist' not in sample.keys() and 'mask' in sample.keys():
            sample_out = {'image': image_out, 'mask': mask_out}
        elif 'dist' not in sample.keys() and 'mask' not in sample.keys():
            sample_out = {'image': image_out}
        return sample_out



