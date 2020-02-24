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
            #print('{}: type: {} dtype: {}'.format(self.__class__.__name__, type(image), image.dtype))
            image = self.transform(image)
            mask = self.transform(mask)
            #print('{}: type: {} dtype: {}'.format(self.__class__.__name__, type(image), image.dtype))
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
    DEFAULT_SEED = 33
    def __init__(self, scale: float, seed=None):
        if not seed:
            seed = self.DEFAULT_SEED
        super(Rescale, self).__init__(p=DEFAULT_TRANSFORM_PROB, seed=seed)
        assert isinstance(scale, float)
        self.output_scale = scale

    def transform(self, tensor: np.array) -> np.array:
        if tensor is None:
            return None
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
    DEFAULT_SEED = 42

    def __init__(self, row_slice=ROW_SLICE, col_slice=COL_SLICE, seed=None):
        if not seed:
            seed = self.DEFAULT_SEED
        super(Crop, self).__init__(p=DEFAULT_TRANSFORM_PROB, seed=seed)
        self.row_slice = row_slice
        self.col_slice = col_slice

    def transform(self, tensor):
        """Slice input tensor with pre-defined slices"""
        if tensor is None:
            return None
        return tensor[self.row_slice, self.col_slice]


class Swap(SampleTransformer):
    DEFAULT_SEED = 27
    def __init__(self, p=RANDOM_TRANSFORM_PROB, seed=None):
        if not seed:
            seed = self.DEFAULT_SEED
        super(Swap, self).__init__(p=p, seed=seed)

    def transform(self, tensor):
        """"""
        if tensor is None:
            return None
        return tensor.swapaxes(0, 1)


class FlipLR(SampleTransformer):
    DEFAULT_SEED = 87

    def __init__(self, p=RANDOM_TRANSFORM_PROB, seed=None):
        if not seed:
            seed = self.DEFAULT_SEED
        super(FlipLR, self).__init__(p=p, seed=seed)

    def transform(self, tensor):
        """"""
        if tensor is None:
            return None
        return np.fliplr(tensor).copy()


class FlipUD(SampleTransformer):
    DEFAULT_SEED = 56

    def __init__(self, p=RANDOM_TRANSFORM_PROB, seed=None):
        if not seed:
            seed = self.DEFAULT_SEED
        super(FlipUD, self).__init__(p=p, seed=seed)

    def transform(self, tensor):
        """"""
        if tensor is None:
            return None
        return np.flipud(tensor).copy()


class RandomCrop(SampleTransformer):
    CROP_CHOICES = [0, 200, 500, 750, 1000]
    ROWS_OFFSET = 1400

    # Default Random Seed
    DEFAULT_SEED = 92
    DEFAULT_CROP_SEED = 43

    def __init__(self, p=DEFAULT_TRANSFORM_PROB, seed=None, crop_seed=None):

        if not seed:
            seed = self.DEFAULT_SEED
        if not crop_seed:
            crop_seed = self.DEFAULT_CROP_SEED

        super(RandomCrop, self).__init__(p=p, seed=seed)
        self._row_crop_choices = np.asarray([a for a in self.CROP_CHOICES])
        self._col_slice = COL_SLICE
        # This is None by default, will be overwritten every time a transformation op is applied.
        self._row_slice = None
        self._random_choice_gen = np.random.RandomState(seed=crop_seed)

    def transform(self, tensor):
        if tensor is None:
            return None
        return tensor[self._row_slice, self._col_slice]
    
    def apply_transform(self, image, mask):
        apply_transform = self._random_state.random_sample()
        if apply_transform < self._transform_p:
            crop_row = self._random_choice_gen.choice(self._row_crop_choices)
            self._row_slice = slice(crop_row, crop_row + self.ROWS_OFFSET)
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


class GaussianNoise(SampleTransformer):
    DEFAULT_SEED = 29
    DEFAULT_NOISE_SEED = 9

    def __init__(self, mean=0, sigma=20, p=RANDOM_TRANSFORM_PROB, seed=None,
                 noise_seed=None):
        if not seed:
            seed=self.DEFAULT_SEED
        if not noise_seed:
            noise_seed=self.DEFAULT_NOISE_SEED

        super(GaussianNoise, self).__init__(p=p, seed=seed)
        self._mean = mean
        self._sigma = sigma
        self._random_noise_gen = np.random.RandomState(seed=noise_seed)

    def transform(self, tensor):
        if tensor is None:
            return None
        row, col, *rest = tensor.shape
        if row == 1:  # 3D, channels first
            _, row, col = tensor.shape
        #sigma = self._random_noise_gen.randint(0, 10000)
        #sigma = 400
        #sigma **= 0.5
        gauss = self._random_noise_gen.normal(self._mean, self._sigma, (row, col))
        #print(np.min(tensor), np.max(tensor), tensor.dtype, gauss.dtype)
        tensor_g = tensor+gauss
        tensor_g /= tensor_g.max()
        #print(np.min(tensor_g), np.max(tensor_g), tensor_g.dtype)
        tensor_g = (tensor_g*255).astype(np.uint8)
        #print(np.min(tensor_g), np.max(tensor_g), tensor_g.dtype)
        return tensor_g

    def apply_transform(self, image, mask):
        apply_transform = self._random_state.random_sample()
        if apply_transform < self._transform_p:
            image = self.transform(image)
        return image, mask
