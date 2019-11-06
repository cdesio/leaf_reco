IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

DATA_DIR_IH = "/data/uob"
DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"

import os
import numpy as np
from UNet_inverted_dice import get_unet

import tensorflow as tf

tf.keras.backend.clear_session()

## Loading data
print("Loading data")
data_dir = os.path.join(DATA_DIR_IH, "Jordan")

X_train = np.load(os.path.join(data_dir, "Xy_train.npz"))["x"]
y_train = np.load(os.path.join(data_dir, "Xy_train.npz"))["y"]

model = get_unet()
model.summary()

model.fit(x=X_train, y=y_train, epochs=200, batch_size=2, verbose=1, validation_split=.2)

model.save(os.path.join(DATA_DIR_IH, "trained_models", "trained_UNet_100epochs_inv_dice.hdf5")
