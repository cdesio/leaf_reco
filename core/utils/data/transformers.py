import torch
from functools import partial
import numpy as np
from skimage.transform import rescale
from abc import abstractmethod, ABC

IMG_WIDTH = 1400
IMG_HEIGHT = 1400
# ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)
ROW_SLICE = slice(1000, 2400)

DEFAULT_TRANSFORM_PROB = 1.0
RANDOM_TRANSFORM_PROB = 0.5
DEFAULT_RANDOM_SEED = 8


class SampleTransformer(ABC):

    def __init__(self, p: float = DEFAULT_TRANSFORM_PROB, seed: int = DEFAULT_RANDOM_SEED):
        self._transform_p = p
        self._random_state = np.random.RandomState()
        self._random_state.seed(seed)

    @abstractmethod
    def transform(self, tensor: np.array) -> np.array:
        raise NotImplementedError

    def apply_transform(self, image, mask):
        apply_transform = self._random_state.random_sample()
        if apply_transform < self._transform_p:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

    def __call__(self, sample):
        if not 'image' in sample.keys():
            return None
        image = sample['image']
        mask = sample['mask'] if 'mask' in sample.keys() else None
        dist = sample['dist'] if 'dist' in sample.keys() else None

        image, mask = self.apply_transform(image, mask)

        sample_out = {'image': image}
        if mask is not None:
            sample_out['mask'] = mask
        if dist is not None:
            sample_out['dist'] = dist
        return sample_out


class ChannelsFirst(SampleTransformer):

    def transform(self, tensor: np.array) -> np.array:
        """
        Apply Channel-first transformation input tensor (as numpy array)

        Cases:
        1. If input tensor is None, None is returned.

        2. If the input tensor is three-dimensional, it is
        transformed with channel-first, unless first dimension
        is already 1.

        3. If input tensor is 2D, a new axis (`np.newaxis`)
        is added as first dimension.

        Parameters:
        tensor: numpy.ndarray

        """
        if tensor is None:
            return None

        if len(tensor.shape) == 3:
            if tensor.shape[0] == 1:
                return tensor
            else:
                return np.transpose(tensor, (2, 0, 1))
        else:
            # 2D image, Add Channel
            return tensor[np.newaxis, ...]


class Rescale(SampleTransformer):

    def __init__(self, scale: float):
        super(Rescale, self).__init__(p=DEFAULT_TRANSFORM_PROB)
        assert isinstance(scale, float)
        self.output_scale = scale

    def transform(self, tensor: np.array) -> np.array:
        """Apply Rescale transformation to input tensor (`numpy.ndarray`)
        """
        resizer = partial(rescale, scale=self.output_scale, anti_aliasing=True)
        if len(tensor.shape) == 3:
            multichannel = True
        else:
            multichannel = False
        return resizer(tensor, multichannel=multichannel)


class ToTensor(SampleTransformer):

    def transform(self, tensor):
        """Transform input tensor into a torch.Tensor object"""
        if tensor is None:
            return None
        return torch.from_numpy(tensor)

    def __call__(self, sample):
        sample_out = super(ToTensor, self).__call__(sample)
        if 'dist' in sample_out:
            dist = sample_out['dist']
            dist_out = torch.from_numpy(np.asarray(dist))
            sample_out['dist'] = dist_out
        return sample_out


class Crop(SampleTransformer):

    def __init__(self, row_slice=ROW_SLICE, col_slice=COL_SLICE):
        super(Crop, self).__init__(p=DEFAULT_TRANSFORM_PROB)
        self.row_slice = row_slice
        self.col_slice = col_slice

    def transform(self, tensor):
        """Slice input tensor with pre-defined slices"""
        if tensor is None:
            return None
        return tensor[self.row_slice, self.col_slice]


class Swap(SampleTransformer):

    def __init__(self, p=RANDOM_TRANSFORM_PROB):
        super(Swap, self).__init__(p=p)

    def transform(self, tensor):
        """"""
        if tensor is None:
            return None
        return tensor.swapaxes(0, 1)


class FlipLR(SampleTransformer):

    def __init__(self, p=RANDOM_TRANSFORM_PROB):
        super(FlipLR, self).__init__(p=p)

    def transform(self, tensor):
        """"""
        if tensor is None:
            return None
        return np.fliplr(tensor)


class FlipUD(SampleTransformer):

    def __init__(self, p=RANDOM_TRANSFORM_PROB):
        super(FlipUD, self).__init__(p=p)

    def transform(self, tensor):
        """"""
        if tensor is None:
            return None
        return np.flipud(tensor)


class RandomCrop(SampleTransformer):
    CROP_CHOICES = [0, 200, 500, 750, 1000]
    COLS_OFFSET = 1400

    def __init__(self, p=RANDOM_TRANSFORM_PROB, seed=DEFAULT_RANDOM_SEED, crop_seed=DEFAULT_RANDOM_SEED):
        super(RandomCrop, self).__init__(p=p)
        self._row_crop_slices = np.asarray([(a, a + self.COLS_OFFSET) for a in self.CROP_CHOICES])
        self._col_slice = COL_SLICE
        self._random_choice_gen = np.random.RandomState(seed=crop_seed)


    def transform(tensor, row_slice, col_slice):
        if tensor is None:
            return None
        return tensor[row_slice, col_slice]

    def apply_transform(self, image, mask):
        apply_transform = self._random_state.random_sample()
        if apply_transform < self._transform_p:
            row_slice = self._random_choice_gen.choice(self._row_crop_slices)
            image = self._crop(image, row_slice, self._col_slice)
            mask = self._crop(mask, row_slice, self._col_slice)
        return image, mask


class GaussianNoise(SampleTransformer):

    def __init__(self, variance, p=RANDOM_TRANSFORM_PROB, noise_seed=DEFAULT_RANDOM_SEED):#seed=DEFAULT_TRANSFORM_PROB, noise_seed=DEFAULT_RANDOM_SEED):
        super(GaussianNoise, self).__init__(p=p)#, seed=seed)
        self.var = variance
        self._random_noise_gen = np.random.RandomState(seed=noise_seed)

    def transform(self, tensor):
        row, col, _ = tensor.shape
        if row == 1:  # 3D, channels first
            _, row, col = tensor.shape
        mean = self._random_noise_gen.randint(0, 30)
        # var = var
        sigma = self.var ** 0.5
        gauss = self._random_noise_gen.normal(mean, sigma, (row, col)).reshape(row, col)
        return tensor + gauss

    def apply_transform(self, image, mask):
        apply_transform = self._random_state.random_sample()
        if apply_transform < self._transform_p:
            image = self.transform(image)
        return image, mask
