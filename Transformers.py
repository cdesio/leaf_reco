from skimage.transform import rescale

import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler

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
            mask_sw1 = mask.swapaxes(2, 0)
            mask_sw2 = mask_sw1.swapaxes(2,1)
        elif len(image.shape) == 2:
            image_newaxis = image[np.newaxis, ...]
            mask_newaxis = mask[np.newaxis, ...]

        if len(sample.keys()) == 3:
            sample_out = {'image': image_sw2, 'mask': mask_sw2, 'dist': dist}
        elif len(sample.keys()) == 2:
            sample_out = {'image': image_newaxis, 'mask': mask_newaxis}
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


def splitter(dataset, validation_split=0.2, batch=16, workers=4):
    dataset_len = len(dataset)
    indices = list(range(dataset_len))
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch, num_workers=workers)
    validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch, num_workers=workers)

    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(train_idx), "val": val_len}
    return data_loaders, data_lengths

def splitter_train_val_test(dataset, validation_split=0.2, test_split=0.2, batch=16, workers=4):
    dataset_len = len(dataset)
    indices = list(range(dataset_len))

    test_len = int(np.floor(test_split * dataset_len))
    train_len = dataset_len - test_len

    test_idx = np.random.choice(indices, size=test_len, replace=False)
    train_idx = list(set(indices) - set(test_idx))

    test_sampler = SubsetRandomSampler(test_idx)

    validation_len = int(np.floor(validation_split * train_len))
    validation_idx = np.random.choice(train_idx, size=validation_len, replace=False)
    train_idx_out = list(set(train_idx)- set(validation_idx))

    validation_sampler = SubsetRandomSampler(validation_idx)
    train_sampler = SubsetRandomSampler(train_idx_out)

    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch, num_workers=workers)
    validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch, num_workers=workers)
    test_loader = DataLoader(dataset, sampler = test_sampler, batch_size=batch, num_workers=workers)
    data_loaders = {"train": train_loader, "val": validation_loader, "test": test_loader}
    data_lengths = {"train": len(train_idx_out), "val": validation_len, "test": test_len}
    return data_loaders, data_lengths




class Cut:

    def __init__(self, cut=True):
        assert isinstance(cut, bool)
        self.cut = cut

    def __call__(self, sample):
        if len(sample.keys()) == 3:
            image, mask, dist = sample['image'], sample['mask'], sample['dist']
        elif len(sample.keys()) == 2:
            image, mask = sample['image'], sample['mask']

        if self.cut:
            out_image = image[ROW_SLICE, COL_SLICE]
            out_mask = mask[ROW_SLICE, COL_SLICE]
        else:
            out_image = image
            out_mask = mask

        if len(sample.keys()) == 3:
            sample_out = {'image': out_image, 'mask': out_mask, 'dist': dist}
        elif len(sample.keys()) == 2:
            sample_out = {'image': out_image, 'mask': out_mask}
        return sample_out