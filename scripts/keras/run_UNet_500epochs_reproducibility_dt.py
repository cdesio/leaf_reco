IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

DATA_DIR_IH = "/data/uob"
DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"

import os
import numpy as np
import keras
from UNet import get_unet
import tensorflow as tf

tf.keras.backend.clear_session()

## Loading data
print("Loading data")

data_dir = DATA_DIR_DEEPTHOUGHT

X_train = np.load(os.path.join(data_dir, "Xy_train.npz"))["x"]
y_train = np.load(os.path.join(data_dir, "Xy_train.npz"))["y"]
model = get_unet()
model.summary()

model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
print("saving model after 100 epochs")
model.save(os.path.join(data_dir, "trained_models", "REPRODUCIBILITY_TEST_trained_UNet_100epochs.hdf5"))
model.fit(x=X_train, y=y_train, epochs=150, batch_size=1, verbose=1, validation_split=.2)
print("after 250")
model.save(os.path.join(data_dir, "trained_models", "REPRODUCIBILITY_TEST_trained_UNet_250epochs.hdf5"))
model.fit(x=X_train, y=y_train, epochs=250, batch_size=1, verbose=1, validation_split=.2)
print("after 500")
model.save(os.path.join(data_dir, "trained_models", "REPRODUCIBILITY_TEST_trained_UNet_500epochs.hdf5"))

model.fit(x=X_train, y=y_train, epochs=500, batch_size=1, verbose=1, validation_split=.2)
print("after 1000")
model.save(os.path.join(data_dir, "trained_models", "REPRODUCIBILITY_TEST_trained_UNet_1000epochs.hdf5"))

model.fit(x=X_train, y=y_train, epochs=500, batch_size=1, verbose=1, validation_split=.2)

model.save(os.path.join(data_dir, "trained_models", "REPRODUCIBILITY_TEST_trained_UNet_1500epochs.hdf5"))
print("saving after 1500")
model.fit(x=X_train, y=y_train, epochs=500, batch_size=1, verbose=1, validation_split=.2)
print("saving after 2000")
model.save(os.path.join(data_dir, "trained_models", "REPRODUCIBILITY_TEST_trained_UNet_2000epochs.hdf5"))
