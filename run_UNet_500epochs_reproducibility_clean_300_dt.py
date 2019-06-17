IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

DATA_DIR_IH="/data/uob"
DATA_DIR_DEEPTHOUGHT="/storage/yw18581/data"

import os
import numpy as np
import keras
from UNet import get_unet
import tensorflow as tf 
from pickle import dump
tf.keras.backend.clear_session()

## Loading data
print("Loading data")

data_dir = DATA_DIR_DEEPTHOUGHT
train_test = os.path.join(data_dir, "train_validation_test"
X_tr = np.load(os.path.join(train_test, "Xy_train_clean_300_24_10_25.npz"))["x"]
y_tr = np.load(os.path.join(train_test,"Xy_train_clean_300_24_10_25.npz"))["y"]

X_val = np.load(os.path.join(train_test, "Xy_val_clean_300_24_10_25.npz"))["x"]
y_val = np.load(os.path.join(train_test,"Xy_val_clean_300_24_10_25.npz"))["y"]

X_train = np.vstack(X_tr, X_val)
y_train = np.vstack(y_tr, y_val)

model = get_unet()
model.summary()

history_100 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
print("saving model after 100 epochs")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_100epochs.hdf5"))
history_250 = model.fit(x=X_train, y=y_train, epochs=150, batch_size=1, verbose=1, validation_split=.2)
print("after 250")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_250epochs.hdf5"))
history_500 = model.fit(x=X_train, y=y_train, epochs=250, batch_size=1, verbose=1, validation_split=.2)
print("after 500")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_500epochs.hdf5"))

history_1000 = model.fit(x=X_train, y=y_train, epochs=500, batch_size=1, verbose=1, validation_split=.2)
print("after 1000")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1000epochs.hdf5"))

history_1500 = model.fit(x=X_train, y=y_train, epochs=500, batch_size=1, verbose=1, validation_split=.2)

model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1500epochs.hdf5"))
print("saving after 1500")
history_2000 = model.fit(x=X_train, y=y_train, epochs=500, batch_size=1, verbose=1, validation_split=.2)
print("saving after 2000")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_2000epochs.hdf5"))

history = list([history_100, history_250, history_500, history_1000, history_1500, history_2000])
dump(history, open(os.path.join(data_dir, "history_REPRODUCIBILITY_TEST_UNet_2000_epochs_clean_300.pkl"),'wb'))
