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


data_dir = "/storage/yw18581/data/"
TRAIN_VAL_TEST_DIR = os.path.join(data_dir, "train_validation_test")

N_EPOCHS=200

CHECKPOINT_FOLDER_PATH = os.path.join(data_dir, 'trained_models')
TASK_NAME = 'CNN_regression_pyimage_{}epochs'.format(N_EPOCHS)
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)


X_train = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_train_dist.npz"))["y"]
X_val = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_val_dist.npz"))["y"]
X_test = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_test_dist.npz"))["y"]

y_train = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_train_dist.npz"))["dist"]
y_val = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_val_dist.npz"))["dist"]
y_test = np.load(os.path.join(TRAIN_VAL_TEST_DIR,"Xy_test_dist.npz"))["dist"]

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
model.fit(X_train_cut, trainY, validation_data=(X_val_cut, valY),
	epochs=N_EPOCHS, batch_size=8, verbose=1)

# make predictions on the testing data
print("[INFO] predicting distances...")
preds = model.predict(X_test_cut)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
