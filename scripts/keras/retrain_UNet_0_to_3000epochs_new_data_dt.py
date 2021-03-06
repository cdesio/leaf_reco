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
from data_loaders_km3 import data_generator_index_list, get_n_iterations_index
from train_test_validation_metadata_arguments_dt import train_validation_test
from network_models import train_neural_network
import tensorflow as tf
from pickle import dump

tf.keras.backend.clear_session()

## Loading data
print("Loading data")

data_dir = DATA_DIR_DEEPTHOUGHT
clean_dir = os.path.join(DATA_DIR_DEEPTHOUGHT, "train_validation_test", "clean_300_june")

CHECKPOINT_FOLDER_PATH = os.path.join(data_dir, 'trained_models')
TASK_NAME = 'UNet_retrain_new_data_clean_300_june'
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

BATCH_SIZE = 1
EPOCHS = 1500


def import_and_split(dist):
    Xy = np.load(os.path.join(clean_dir, "Xy_{}mm_clean300_june.npz".format(dist)))
    X = Xy["x"]
    y = Xy["y"]
    _, test_indices, train_indices, val_indices = train_validation_test(X)
    Xy.close()
    return X, y, train_indices, val_indices, test_indices


def export_indices(dist_list):
    for d in dist_list:
        print("loading {}mm".format(d))
        _, _, train_indices_d, val_indices_d, test_indices_d = import_and_split(d)
        print("saving indices for {}mm".format(d))
        np.savez_compressed(os.path.join(TASK_FOLDER_PATH, "train_val_test_indices_{}mm.npz".format(d)),
                            train=train_indices_d, val=val_indices_d, test=test_indices_d)
    return


distances = list([1, 2, 4, 10, 25, 35])

export_indices(distances)


def load_indices(d, key):
    indices = np.load(os.path.join(TASK_FOLDER_PATH, "train_val_test_indices_{}mm.npz".format(d)))
    if key == "train":
        return indices["train"]
    elif key == "val":
        return indices["val"]


fnames_list = [os.path.join(clean_dir, "Xy_{}mm_clean_300.npz".format(d)) for d in distances]
index_list_train = [load_indices(d, "train") for d in distances]
index_list_val = [load_indices(d, "val") for d in distances]
print("creating data generators")

train_generator = data_generator_index_list(fnames_list, index_list=index_list_train, batch_size=BATCH_SIZE,
                                            ftarget=lambda y: y, )
validation_generator = data_generator_index_list(fnames_list, index_list=index_list_val, batch_size=BATCH_SIZE,
                                                 ftarget=lambda y: y, )
print("calculating steps per epoch")
steps_per_epoch, n_events = get_n_iterations_index(fnames_list, index_list_train, batch_size=BATCH_SIZE)
validation_steps, n_events = get_n_iterations_index(fnames_list, index_list_val, batch_size=BATCH_SIZE)
##X_train = np.load(os.path.join(data_dir,"Xy_train.npz"))["x"]
# y_train = np.load(os.path.join(data_dir,"Xy_train.npz"))["y"]
model = get_unet()
model.summary()
print("training model and save")


def train_and_save(nn_model, n_epochs):
    history = nn_model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs,
                                     validation_data=validation_generator, validation_steps=validation_steps,
                                     verbose=verbose, use_multiprocessing=True)

    nn_model.save(os.path.join(TASK_FOLDER_PATH, "retrained_UNet_{}_epochs_clean_300_june.hdf5".format(n_epochs)))
    dump(history,
         open(os.path.join(TASK_FOLDER_PATH, "history_retrained_UNet_{}_epochs_clean_300_june.pkl".format(n_epochs)),
              'wb'))
    return nn_model


model_250 = train_and_save(model, 250)
model_500 = train_and_save(model_250, 250)
model_750 = train_and_save(model_500, 250)
model_1000 = train_and_save(model_750, 250)
model_1500 = train_and_save(model_1000, 500)
model_2000 = train_and_save(model_1500, 500)
model_2500 = train_and_save(model_2000, 500)
model_3000 = train_and_save(model_2500, 500)
print("done.")
# model.load_weights(os.path.join(data_dir,"trained_models/retrained_UNet_500+250+250epochs.hdf5"))

# model.fit(x=X_train, y=y_train, epochs=250, batch_size=1, verbose=1, validation_split=.2)
# print("saving trained model")
# model.save(os.path.join(TASK_FOLDER_PATH,"retrained_UNet_1500_epochs_clean_300_batch1.hdf5"))

# model.fit(x=X_train, y=y_train, epochs=250, batch_size=1, verbose=1, validation_split=.2)
# model.save(os.path.join(data_dir,"trained_models","retrained_UNet_1000+500epochs.hdf5"))
