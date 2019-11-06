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
train_test = os.path.join(data_dir, "train_validation_test")
X_tr = np.load(os.path.join(train_test, "Xy_train_clean_300_24_10_25.npz"))["x"]
y_tr = np.load(os.path.join(train_test,"Xy_train_clean_300_24_10_25.npz"))["y"]

X_val = np.load(os.path.join(train_test, "Xy_val_clean_300_24_10_25.npz"))["x"]
y_val = np.load(os.path.join(train_test,"Xy_val_clean_300_24_10_25.npz"))["y"]

X_train = np.vstack((X_tr, X_val))
y_train = np.vstack((y_tr, y_val))

model = get_unet()
model.summary()

#history_100 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
#print("saving model after 100 epochs")
#model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_100epochs.hdf5"))
#history_250 = model.fit(x=X_train, y=y_train, epochs=150, batch_size=1, verbose=1, validation_split=.2)
#print("after 250")
#model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_250epochs.hdf5"))
#history_500 = model.fit(x=X_train, y=y_train, epochs=250, batch_size=1, verbose=1, validation_split=.2)
#print("after 500")
model.load_weights(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1100epochs.hdf5"))
#history_600 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
#print("after 600")
#model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_600epochs.hdf5"))
#history_700 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
#print("after 700")
#model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_700epochs.hdf5"))
#history_800 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
#print("after 800")
#model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_800epochs.hdf5"))
#history_900 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
#print("after 900")
#model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_900epochs.hdf5"))
#history_1000 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
#print("after 1000")
#model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1000epochs.hdf5"))
#history_1100 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
#print("after 1100")
#model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1100epochs.hdf5"))
history_1200 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)

model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1200epochs.hdf5"))
print("saving after 1200")
history_1300 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)

model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1300epochs.hdf5"))
print("saving after 1300")
history_1400 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)

model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1400epochs.hdf5"))
print("saving after 1400")
history_1500 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
print("saving after 1500")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1500epochs.hdf5"))
history_1600 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
print("saving after 1600")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1600epochs.hdf5"))
history_1700 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
print("saving after 1700")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1700epochs.hdf5"))
history_1800 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
print("saving after 1800")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1800epochs.hdf5"))
history_1900 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
print("saving after 1900")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_1900epochs.hdf5"))
history_2000 = model.fit(x=X_train, y=y_train, epochs=100, batch_size=1, verbose=1, validation_split=.2)
print("saving after 2000")
model.save(os.path.join(data_dir,"trained_models","REPRODUCIBILITY_TEST_clean_300_trained_UNet_2000epochs.hdf5"))

#history = list([history_100, history_250, history_500, history_1000, history_1500, history_2000])
#dump(history, open(os.path.join(data_dir, "history_REPRODUCIBILITY_TEST_UNet_2000_epochs_clean_300.pkl"),'wb'))
