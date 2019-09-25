from torch.utils.data import Dataset, DataLoader

from matplotlib.image import imread

import re
import os

IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)


class UNetDataSetFromNpz(Dataset):
    def __init__(self, X, Y, transform=None, dist=None):
        self.transform = transform
        self._X = X
        self._Y = Y
        self._dist = dist

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

class UNetDatasetFromFolders(Dataset):

    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.root_path = root_path
        self.distances, self.images_list, self.masks_list = self._create_list()

    @staticmethod
    def file_sort_key(fpath: str):
        # os.path.split(fpath) --> base, fname
        # os.path.splitext(fname) --> name, ext
        # example_filename: File_3_1mm_mask_2155.tiff
        _, fname = os.path.split(fpath)
        fname, _ = os.path.splitext(fname)
        _, dist, *rest = fname.split('_')
        return int(dist)

    def _create_list(self):
        regex = re.compile(r'\d+')
        distances = []
        images_list = []
        masks_list = []

        for root_dir, _, files in os.walk(self.root_path):
            image_found = 0
            mask_found = 0
            folder_imgs = []
            folder_masks = []

            for fname in files:
                if fname.startswith("File") and fname.endswith('.tiff'):
                    if "mask" not in fname:
                        image_found += 1
                        folder_imgs.append(os.path.join(root_dir, fname))
                    elif 'mask' in fname:
                        mask_found+=1
                        folder_masks.append(os.path.join(root_dir, fname))

            assert len(folder_imgs) == len(folder_masks)
            assert image_found==mask_found

            folder_imgs = sorted(folder_imgs, key=self.file_sort_key)
            folder_masks = sorted(folder_masks, key=self.file_sort_key)

            images_list.extend(folder_imgs)
            masks_list.extend(folder_masks)


            if image_found or mask_found:
                folder = root_dir.split('/')[-1]
                dist = regex.findall(folder.split('_')[1])[0]
                distances.extend(int(dist) for _ in range(image_found))

        return distances, images_list, masks_list

    def __getitem__(self, idx):
        image = imread(self.images_list[idx])
        mask = imread(self.masks_list[idx])
        distance = self.distances[idx]
        sample = {'image': image, 'mask': mask, 'dist': distance}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images_list)
