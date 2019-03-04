IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
from UNet import get_unet
from data_loaders_km3 import data_generator, get_n_iterations
from network_models_km3 import train_neural_network
from os import path as p
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

tf.keras.backend.clear_session()

DATA_DIR_IH="/data/uob"
DATA_DIR_DEEPTHOUGHT="/storage/yw18581/data"

data_folder = DATA_DIR_IH
TRAIN_VAL_TEST_DIR = os.path.join(data_folder,"train_validation_test")


N_FILES = 1
BATCH_SIZE=1
N_EPOCHS = 500

model = get_unet(compile_model=True)
model.summary()


CHECKPOINT_FOLDER_PATH = p.join(data_folder, 'trained_models')
TASK_NAME = 'UNet_training_generator_{}epochs'.format(N_EPOCHS)
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

TRAINING_WEIGHTS_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                         '{}_weights_training{}.hdf5'.format(model.name, TASK_NAME))

HISTORY_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                '{}_history{}.pkl'.format(model.name, TASK_NAME))

MODEL_JSON_FILEPATH = os.path.join(TASK_FOLDER_PATH, '{}.json'.format(model.name))



fname_train = [os.path.join(TRAIN_VAL_TEST_DIR,"Xy_train.npz")]
fname_val = [os.path.join(TRAIN_VAL_TEST_DIR,"Xy_val.npz")]

steps_per_epoch, n_events = get_n_iterations(fname_train, batch_size=BATCH_SIZE)
print("training steps per epoc:{}, number of events:{}".format(steps_per_epoch, n_events))

validation_steps, n_evts_val = get_n_iterations(fname_val, batch_size=BATCH_SIZE)
print("validation steps per epoch:{}, number of events:{}".format(validation_steps, n_evts_val))

#prediction_steps, n_evts_test = get_n_iterations(fname_test, batch_size=BATCH_SIZE)
#print(prediction_steps, n_evts_test)

training_generator = data_generator(fname_train, batch_size=BATCH_SIZE,
                                    ftarget=lambda y: y)

validation_generator = data_generator(fname_val, batch_size=BATCH_SIZE,
                                     ftarget=lambda y: y)
#data_dir = DATA_DIR_IH

#model = multi_gpu_model(model, gpus=2)
#model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
training_history = train_neural_network(model, training_generator, steps_per_epoch,
                                        validation_generator, validation_steps, 
                                        no_stopping=True,
                                        batch_size=BATCH_SIZE, epochs=N_EPOCHS)


print('Saving Model (JSON), Training History & Weights...', end='')
model_json_str = model.to_json()
with open(MODEL_JSON_FILEPATH, 'w') as model_json_f:
    model_json_f.write(model_json_str)

history_filepath = HISTORY_FILEPATH
dump(training_history.history, open(history_filepath, 'wb'))

model.save_weights(TRAINING_WEIGHTS_FILEPATH)
print('...Done!')
