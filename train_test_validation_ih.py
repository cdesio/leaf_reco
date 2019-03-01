IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split


DATA_DIR_IH="/data/uob"
DATA_DIR_DEEPTHOUGHT="/storage/yw18581/data"

data_dir = DATA_DIR_IH
source_data_folder = os.path.join(data_dir,'Jordan')
TRAIN_VAL_TEST_DIR = os.path.join(data_dir,"train_validation_test")

if not os.path.exists(TRAIN_VAL_TEST_DIR):
    os.makedirs(TRAIN_VAL_TEST_DIR)



## Loading data
print("Loading data filenames")


folder_2mm=os.path.join(source_data_folder,"10x10_2mm_8bit"
fnames_orig_2mm = [os.path.join(folder_2mm,"/{}".format(filename)) for filename in sorted(folder_2mm))) 
               if "mask" not in filename and filename.startswith('File')]
fnames_orig_4mm = [os.path.join(data_dir,"10x10_4mm_v2 copy/{}".format(filename)) 
           for filename in sorted(os.listdir(os.path.join(data_dir,"10x10_4mm_v2 copy/"))) 
               if "mask" not in filename and filename.startswith('File')]
fnames_orig_10mm = [os.path.join(data_dir,"10x10_10mm_v2_8bit/{}".format(filename))
               for filename in sorted(os.listdir(os.path.join(data_dir,"10x10_10mm_v2_8bit/"))) 
               if "mask" not in filename and filename.startswith('File')]
fnames_orig_25mm = [os.path.join(data_dir,"10x10_25mm_8bit/{}".format(filename)) 
               for filename in sorted(os.listdir(os.path.join(data_dir,"10x10_25mm_8bit/"))) 
               if "mask" not in filename and filename.startswith('File')]

fnames_mask_2mm = [os.path.join(data_dir,"10x10_2mm_8bit/{}".format(filename)) 
               for filename in sorted(os.listdir(os.path.join(data_dir,"10x10_2mm_8bit/"))) 
               if "mask" in filename and filename.startswith('File')]
fnames_mask_4mm = [os.path.join(data_dir,"10x10_4mm_v2 copy/{}".format(filename)) 
               for filename in sorted(os.listdir(os.path.join(data_dir,"10x10_4mm_v2 copy/"))) 
               if "mask" in filename and filename.startswith('File')]
fnames_mask_10mm = [os.path.join(data_dir,"10x10_10mm_v2_8bit/{}".format(filename)) 
               for filename in sorted(os.listdir(os.path.join(data_dir,"10x10_10mm_v2_8bit/"))) 
               if "mask" in filename and filename.startswith('File')]
fnames_mask_25mm = [os.path.join(data_dir,"10x10_25mm_8bit/{}".format(filename)) 
               for filename in sorted(os.listdir(os.path.join(data_dir,"10x10_25mm_8bit/"))) 
               if "mask" in filename and filename.startswith('File')]

print("check number of files per type")
print(len(fnames_mask_2mm), len(fnames_mask_4mm), len(fnames_mask_10mm), len(fnames_mask_25mm))

print(len(fnames_orig_2mm), len(fnames_orig_4mm), len(fnames_orig_10mm), len(fnames_orig_25mm))


fnames_mask = np.hstack((fnames_mask_2mm,fnames_mask_4mm, fnames_mask_10mm, fnames_mask_25mm))

fnames_orig = np.hstack((fnames_orig_2mm,fnames_orig_4mm, fnames_orig_10mm, fnames_orig_25mm))
print("check total number of files")
print("original images:{}, mask files:{}".format(len(fnames_orig),len(fnames_mask) ))
print("Import data (this may take some time)")

X = np.asarray([imread(img)[ROW_SLICE, COL_SLICE] for img in fnames_orig])
y = np.asarray([imread(img)[ROW_SLICE, COL_SLICE] for img in fnames_mask])
print("Create X(shape:{}) and y(shape:{})".format(X.shape, y.shape))

print("Train test split")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print("Add new dimension - for channels last data format in keras")
X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
y_test = y_test[...,np.newaxis]

print("Split train dataset into train and validation datasets")

X_train_v, X_val, y_train_v, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train_v.shape, y_train_v.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)


print("Save train, validation and test data to output files in {} , {} and {}".format(os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                              "Xy_train.npz"),
                                                                                 os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                              "Xy_val.npz"),
                                                                                 os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                              "Xy_test.npz")))



np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_train.npz"),
                            x=X_train_v, y=y_train_v)
np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_test.npz"),
                            x=X_test, y=y_test)
np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_val.npz"),
                            x=X_val, y=y_val)
