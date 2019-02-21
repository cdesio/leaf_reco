IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split


## Loading data
print("Loading data filenames")
data_dir = "/data/uob/Jordan"
fnames_orig_2mm = [os.path.join(data_dir,"10x10_2mm_8bit/{}".format(filename))
               for filename in sorted(os.listdir(os.path.join(data_dir,"10x10_2mm_8bit/"))) 
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print("Add new dimension - for channels last data format in keras")
X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]

X_test = X_test[..., np.newaxis]
print("Save train and test data to output files in {} and {}".format(os.path.join(data_dir,"Xy_train.npz"),os.path.join(data_dir,"Xy_test.npz")))
 
      
np.savez_compressed(os.path.join(data_dir,"Xy_train.npz"),
                            x=X_train, y=y_train)
np.savez_compressed(os.path.join(data_dir,"Xy_test.npz"),
                            x=X_test, y=y_test)