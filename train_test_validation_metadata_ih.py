IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
import pandas as pd
import re
regex = re.compile(r'\d+')

DATA_DIR_IH="/data/uob"
DATA_DIR_DEEPTHOUGHT="/storage/yw18581/data"
data_dir = DATA_DIR_IH

source_data_folder = os.path.join(data_dir,"Jordan")
print(data_dir)
print(source_data_folder)
TRAIN_VAL_TEST_DIR = os.path.join(data_dir,"train_validation_test")

if not os.path.exists(TRAIN_VAL_TEST_DIR):
    os.makedirs(TRAIN_VAL_TEST_DIR)


## Loading data
print("Loading data filenames")


folder_2mm=os.path.join(source_data_folder,"10x10_2mm_8bit")
folder_4mm=os.path.join(source_data_folder,"10x10_4mm_v2 copy")
folder_10mm=os.path.join(source_data_folder,"10x10_10mm_v2_8bit")
folder_25mm=os.path.join(source_data_folder,"10x10_25mm_8bit")

def get_filename_and_data(folder):
    images_list = [[os.path.join(folder,"{}".format(filename)), regex.findall(folder)[2]]
                   for filename in sorted(os.listdir(folder)) if "mask" not in filename and filename.startswith("File")]
    masks_list = [[os.path.join(folder,"{}".format(filename)),
                   regex.findall(folder)[2]]
                  for filename in sorted(os.listdir(folder)) if "mask" in filename and filename.startswith("File")]
    df_images = pd.DataFrame.from_records(images_list, columns=('path', 'dist'))
    df_masks = pd.DataFrame.from_records(masks_list, columns=('path', 'dist'))
    return df_images, df_masks

fnames_orig_2mm, fnames_mask_2mm = get_filename_and_data(folder_2mm)
fnames_orig_4mm, fnames_mask_4mm = get_filename_and_data(folder_4mm)
fnames_orig_10mm, fnames_mask_10mm = get_filename_and_data(folder_10mm)
fnames_mask_10mm['dist']=10
fnames_orig_25mm, fnames_mask_25mm = get_filename_and_data(folder_25mm)

print("check number of files per type")
print(len(fnames_mask_2mm), len(fnames_mask_4mm), len(fnames_mask_10mm), len(fnames_mask_25mm))
print(len(fnames_orig_2mm), len(fnames_orig_4mm), len(fnames_orig_10mm), len(fnames_orig_25mm))


df_mask = pd.concat([fnames_mask_2mm,fnames_mask_4mm, fnames_mask_10mm, fnames_mask_25mm], ignore_index=True)
df_orig = pd.concat([fnames_orig_2mm,fnames_orig_4mm, fnames_orig_10mm, fnames_orig_25mm], ignore_index=True)
print("check total number of files")
print("original images:{}, mask files:{}".format(len(df_orig),len(df_mask) ))

print("Train test split")

def train_validation_test(df, stratify=False, stratification_key=None):
    if stratify:
        df_out = df[stratification_key]
    else:
        df_out = df

    indices = np.arange(len(df_out))
    train_indices, test_indices = train_test_split(indices)
    return train_indices, test_indices

train_indices, test_indices = train_validation_test(df_mask, stratify=True, stratification_key='dist')

df_orig_train = df_orig.iloc[train_indices]
df_orig_test = df_orig.iloc[test_indices]

df_mask_train = df_mask.iloc[train_indices]
df_mask_test = df_mask.iloc[test_indices]

print("Import data (this may take some time)")

X_train = np.asarray([imread(df_orig['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in train_indices])
X_test = np.asarray([imread(df_orig['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in test_indices])
y_train = np.asarray([imread(df_mask['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in train_indices])
y_test = np.asarray([imread(df_mask['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in test_indices])


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print("Add new dimension - for channels last data format in keras")
X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
y_test = y_test[...,np.newaxis]

print("Split train dataset into train and validation datasets")


X_train_v, X_val, y_train_v, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Train, validation, test data shape")

print(X_train_v.shape, y_train_v.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)


print("Save train, validation and test data to output files in {} , {} and {}".format(os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                              "Xy_train.npz"),
                                                                                 os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                              "Xy_val.npz"),
                                                                                 os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                              "Xy_test.npz")))

np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_train_strarified_dist.npz"),
                            x=X_train_v, y=y_train_v)
np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_test_strarified_dist.npz"),
                            x=X_test, y=y_test)
np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_val_strarified_dist.npz"),
                            x=X_val, y=y_val)

