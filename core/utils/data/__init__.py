import os
import re
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from .dataset import UNetDatasetFromFolders, UNetDataSetFromNpz
from .transformers import ChannelsFirst, Rescale, ToTensor, Cut


def define_dataset(root_folder, batch_size=16, validation_split=0.2, test_split=0.2,
                   excluded_list=None, include_list=None, scale=0.25, multi_processing=0, alldata=False):
    excluded = excluded_list
    include = include_list
    composed = transforms.Compose([Cut(), Rescale(scale), ChannelsFirst(), ToTensor()])
    dataset = UNetDatasetFromFolders(root_folder, fname_key='File', file_extension='.tiff', excluded=excluded, included=include, transform=composed, fname_key='File')
    if alldata:
        data_loaders = DataLoader(dataset, batch_size=batch_size, num_workers=multi_processing)
        data_lengths = len(dataset)
    else:
        data_loaders, data_lengths = splitter_train_val_test(dataset,
                                                             validation_split,
                                                             test_split,
                                                             batch=batch_size,
                                                             workers=multi_processing)
    return data_loaders, data_lengths


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
    train_idx_out = list(set(train_idx) - set(validation_idx))

    validation_sampler = SubsetRandomSampler(validation_idx)
    train_sampler = SubsetRandomSampler(train_idx_out)

    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch, num_workers=workers)
    validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch, num_workers=workers)
    test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch, num_workers=workers)
    data_loaders = {"train": train_loader, "val": validation_loader, "test": test_loader}
    data_lengths = {"train": len(train_idx_out), "val": validation_len, "test": test_len}
    return data_loaders, data_lengths


def select_dist(dist_list, root_folder):
    regex = re.compile(r'\d+')

    def hasNumbers(inputString):
        return bool(re.search(r'\d', inputString))

    excluded = []
    for dist in dist_list:
        for root, dirs, _ in os.walk(root_folder):
            for d in dirs:
                if hasNumbers(d):
                    if int(regex.findall(d)[-1]) == dist:
                        excluded.append(d)
    return excluded
