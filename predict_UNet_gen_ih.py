IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
from UNet import get_unet
from data_loaders_km3 import data_generator, get_n_iterations
from os import path as p
import tensorflow as tf
from tqdm import tqdm
import numpy as np

tf.keras.backend.clear_session()

DATA_DIR_IH="/data/uob"
DATA_DIR_DEEPTHOUGHT="/storage/yw18581/data"

data_folder = DATA_DIR_IH
TRAIN_VAL_TEST_DIR = os.path.join(data_folder,"train_validation_test")

N_FILES = 1
BATCH_SIZE=15
N_EPOCHS = 500

#model = get_unet()
#model.summary()


CHECKPOINT_FOLDER_PATH = p.join(data_folder, 'trained_models')
TASK_NAME = 'UNet_training_generator_{}epochs'.format(N_EPOCHS)
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

#TRAINING_WEIGHTS_FILEPATH = os.path.join(TASK_FOLDER_PATH,
#                                         '{}_weights_training{}.hdf5'.format(model.name, TASK_NAME))

TRAINING_WEIGHTS_FILEPATH=os.path.join(CHECKPOINT_FOLDER_PATH,'retrained_UNet_500+250epochs.hdf5')

fname_test = [os.path.join(TRAIN_VAL_TEST_DIR,"Xy_test.npz")]

#model.load_weights(TRAINING_WEIGHTS_FILEPATH)
prediction_steps, n_evts_test = get_n_iterations(fname_test, batch_size=BATCH_SIZE)
print("prediction steps per epoch:{}, n events:{}".format(prediction_steps, n_evts_test))

print('INFERENCE STEP')

test_data_gen = data_generator(fname_test, batch_size=BATCH_SIZE,
                               ftarget=lambda y: y)

def inference_step(network_model, test_data_generator, predict_steps):

    y_pred = list()

    for _ in tqdm(range(predict_steps)):
        X_batch, _ = next(test_data_generator)
        with tf.device('/device:CPU:0'):
            Y_batch_pred = network_model.predict_on_batch(X_batch)
        y_pred.append(Y_batch_pred)
    y_pred = np.concatenate(y_pred, axis=0)

    return y_pred

with tf.device('/device:CPU:0'):
    model = get_unet()
    model.load_weights(TRAINING_WEIGHTS_FILEPATH)
    y_pred = list()
    for _ in tqdm(range(prediction_steps)):
        X_batch, _ = next(test_data_gen)
        Y_batch_pred = model.predict_on_batch(X_batch)
        y_pred.append(Y_batch_pred)
    y_pred = np.concatenate(y_pred, axis=0)
    #y_pred = inference_step(model, test_data_gen, prediction_steps)

np.savez_compressed(os.path.join(TASK_FOLDER_PATH,"TEST.npz"),
                            x=np.load(fname_test[0])['x'] , y=y_pred)
