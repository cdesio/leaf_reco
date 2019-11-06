IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
import sys
import re
import numpy as np
from matplotlib.image import imread

regex = re.compile(r'\d+')

if not len(sys.argv) == 3:
    raise Exception("Please insert folder name and out_str")

DATA_DIR_IH = "/data/uob"
DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"
data_dir = DATA_DIR_DEEPTHOUGHT

source_data_folder = data_dir
TRAIN_VAL_TEST_DIR = os.path.join(data_dir, "train_validation_test")

if not os.path.exists(TRAIN_VAL_TEST_DIR):
    os.makedirs(TRAIN_VAL_TEST_DIR)

## Loading data
print("Loading data filenames")
folder = os.path.join(source_data_folder, sys.argv[1])
print("read data from {}".format(folder))
# folder_15mm = os.path.join(source_data_folder, "10x10_15mm_v2_8bit")

from train_test_validation_metadata_dt import get_filename_and_data

df_orig, df_mask = get_filename_and_data(folder)

print("check number of files")
print(len(df_orig), len(df_mask))

X = np.asarray([imread(df_orig['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in range(len(df_orig))])

y = np.asarray([imread(df_mask['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in range(len(df_mask))])

metadata = np.asarray(df_orig['dist'], dtype=int)

print("data loaded")

print(X.shape, y.shape, metadata.shape)

print("Add new dimension - for channels last data format in keras")
X = X[..., np.newaxis]
y = y[..., np.newaxis]

out_str = sys.argv[2]
print("Save data to output file in {}".format(os.path.join(TRAIN_VAL_TEST_DIR, out_str)))

np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR, out_str),
                    x=X, y=y, dist=metadata)
