IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
import numpy as np
from UNet_inverted_dice import get_unet
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.keras.backend.clear_session()

## Loading data
print("Loading data")

fnames_orig_2mm = ["/storage/yw18581/data/10x10_2mm_8bit/{}".format(filename) 
               for filename in sorted(os.listdir("/storage/yw18581/data/10x10_2mm_8bit/")) 
               if "mask" not in filename and filename.startswith('File')]
fnames_orig_4mm = ["/storage/yw18581/data/10x10_4mm_v2 copy/{}".format(filename) 
           for filename in sorted(os.listdir("/storage/yw18581/data/10x10_4mm_v2 copy/")) 
               if "mask" not in filename and filename.startswith('File')]
fnames_orig_10mm = ["/storage/yw18581/data/10x10_10mm_v2_8bit/{}".format(filename) 
               for filename in sorted(os.listdir("/storage/yw18581/data/10x10_10mm_v2_8bit/")) 
               if "mask" not in filename and filename.startswith('File')]
fnames_orig_25mm = ["/storage/yw18581/data/10x10_25mm_8bit/{}".format(filename) 
               for filename in sorted(os.listdir("/storage/yw18581/data/10x10_25mm_8bit/")) 
               if "mask" not in filename and filename.startswith('File')]

fnames_mask_2mm = ["/storage/yw18581/data/10x10_2mm_8bit/{}".format(filename) 
               for filename in sorted(os.listdir("/storage/yw18581/data/10x10_2mm_8bit/")) 
               if "mask" in filename and filename.startswith('File')]
fnames_mask_4mm = ["/storage/yw18581/data/10x10_4mm_v2 copy/{}".format(filename) 
               for filename in sorted(os.listdir("/storage/yw18581/data/10x10_4mm_v2 copy/")) 
               if "mask" in filename and filename.startswith('File')]
fnames_mask_10mm = ["/storage/yw18581/data/10x10_10mm_v2_8bit/{}".format(filename) 
               for filename in sorted(os.listdir("/storage/yw18581/data/10x10_10mm_v2_8bit/")) 
               if "mask" in filename and filename.startswith('File')]
fnames_mask_25mm = ["/storage/yw18581/data/10x10_25mm_8bit/{}".format(filename) 
               for filename in sorted(os.listdir("/storage/yw18581/data/10x10_25mm_8bit/")) 
               if "mask" in filename and filename.startswith('File')]


fnames_mask = np.hstack((fnames_mask_2mm,fnames_mask_4mm, fnames_mask_10mm, fnames_mask_25mm))

fnames_orig = np.hstack((fnames_orig_2mm,fnames_orig_4mm, fnames_orig_10mm, fnames_orig_25mm))
print("check number of files")
print("original images:{}, mask files:{}".format(len(fnames_orig),len(fnames_mask) ))

X = np.asarray([imread(img)[ROW_SLICE, COL_SLICE] for img in fnames_orig])
y = np.asarray([imread(img)[ROW_SLICE, COL_SLICE] for img in fnames_mask])
print("Create X(shape:{}) and y(shape:{})".format(X.shape, y.shape))

print("Train test split")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]

X_test = X_test[..., np.newaxis]
#y_test = y_test[..., np.newaxis]


model = get_unet()
model.summary()

model.fit(x=X_train, y=y_train, epochs=200, batch_size=2, verbose=1, validation_split=.2)

model.save("/storage/yw18581/data/trained_UNet_100epochs_inv_dice.hdf5")
sys.exit()
