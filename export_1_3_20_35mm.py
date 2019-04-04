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
data_dir = DATA_DIR_DEEPTHOUGHT

source_data_folder = data_dir
print(data_dir)
print(source_data_folder)
TRAIN_VAL_TEST_DIR = os.path.join(data_dir, "train_validation_test")


if not os.path.exists(TRAIN_VAL_TEST_DIR):
    os.makedirs(TRAIN_VAL_TEST_DIR)


## Loading data
print("Loading data filenames")

folder_15mm = os.path.join(source_data_folder, "10x10_15mm_v2_8bit")

from train_test_validation_metadata_dt import get_filename_and_data


df_orig, df_mask = get_filename_and_data(folder_15mm)

print("check number of files") 
print(len(df_orig), len(df_mask))

X_15mm = np.asarray([imread(df_orig['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in range(len(df_orig))])

y_15mm = np.asarray([imread(df_mask['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in range(len(df_mask))])

metadata = np.asarray(df_orig['dist'], dtype=int)


print(X_15mm.shape, y_15mm.shape, metadata.shape)

print("Add new dimension - for channels last data format in keras")
X_15mm = X_15mm[..., np.newaxis]
y_15mm = y_15mm[..., np.newaxis]

out_str = 'Xy_15mm.npz'
print("Save data to output file in {}".format(os.path.join(TRAIN_VAL_TEST_DIR, out_str)))

np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR, out_str),
                        x=X_15mm, y=y_15mm, dist=metadata)

  
