# %load cnn_regression-Copy1.py
# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import models
import numpy as np
import argparse
import locale
import os
import sys
from matplotlib.image import imread
import matplotlib.pyplot as plt
import tensorflow as tf
from pickle import dump

IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)


def cut_X(arr, reshape = None):
    x_cut = arr[:,960:1300,600:]
    if reshape:
        if len(x_cut.shape)>3:
            x_cut = x_cut[...,0]
            x_cut_out = x_cut.reshape(x_cut.shape[0],x_cut.shape[1]*x_cut.shape[2])
    else:
        x_cut_out = x_cut
    return x_cut_out


#data_dir = "/storage/yw18581/data/"
data_dir = "/data/uob"
TRAIN_VAL_TEST_DIR = os.path.join(data_dir, "train_validation_test")

N_EPOCHS=1000

CHECKPOINT_FOLDER_PATH = os.path.join(data_dir, 'trained_models')
TASK_NAME = 'CNN_regression_pyimage_{}epochs'.format(N_EPOCHS)
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)


X_train = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_train_dataset_2_4_15_25_35.npz"))["y"]
X_val = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_val_dataset_2_4_15_25_35.npz"))["y"]
X_test = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_test_dataset_2_4_15_25_35.npz"))["y"]

y_train = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_train_dataset_2_4_15_25_35.npz"))["dist"]
y_val = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_val_dataset_2_4_15_25_35.npz"))["dist"]
y_test = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_test_dataset_2_4_15_25_35.npz"))["dist"]

X_train_cut = cut_X(X_train)
X_val_cut = cut_X(X_val)
X_test_cut = cut_X(X_test)


# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxDist = np.max(y_train)
print(maxDist)
trainY = y_train/maxDist
valY = y_val/maxDist
testY = y_test/maxDist

# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss, implying that we
# seek to minimize the absolute percentage difference between our
# price *predictions* and the *actual prices*

_, width, height, depth,  = X_train_cut.shape

model = models.create_cnn(height,width, depth, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / N_EPOCHS)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.summary()

# train the model
print("[INFO] training model...")
training_history = model.fit(X_train_cut, trainY, validation_data=(X_val_cut, valY),
	epochs=N_EPOCHS, batch_size=8, verbose=1)


print('Saving Model (JSON), Training History & Weights...', end='')
model_json_str = model.to_json()
with open(os.path.join(TASK_FOLDER_PATH,"model_cnn_pyimage_trained.json"), 'w') as model_json_f:
    model_json_f.write(model_json_str)

history_filepath = os.path.join(TASK_FOLDER_PATH, "model_cnn_pyimage_history.pkl")
dump(training_history.history, open(history_filepath, 'wb'))

model.save_weights(os.path.join(TASK_FOLDER_PATH, "model_cnn_pyimage_history.hdf5"))

