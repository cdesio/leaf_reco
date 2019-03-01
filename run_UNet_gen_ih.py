IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
from UNet import get_unet
from data_loaders_km3 import data_generator, get_n_iterations
from os import path as p
import tensorflow as tf


tf.keras.backend.clear_session()

DATA_DIR_IH="/data/uob"
DATA_DIR_DEEPTHOUGHT="/storage/yw18581/data"

data_folder = DATA_DIR_IH
TRAIN_VAL_TEST_DIR = os.path.join(data_folder,"train_validation_test")


N_FILES = 1
BATCH_SIZE=3

CHECKPOINT_FOLDER_PATH = p.join(p.abspath(p.curdir), 'model')
TASK_NAME = 'UNet_training_generator_10epochs'
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

TRAINING_WEIGHTS_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                         '{}_weights_training{}.hdf5'.format(model.name, TASK_NAME))

HISTORY_FILEPATH = os.path.join(TASK_FOLDER_PATH,
                                '{}_history{}.pkl'.format(model.name, TASK_NAME))

MODEL_JSON_FILEPATH = os.path.join(TASK_FOLDER_PATH, '{}.json'.format(model.name))



fname_train = os.path.join(TRAIN_VAL_TEST_DIR,"Xy_train.npz")
fname_val = os.path.join(TRAIN_VAL_TEST_DIR,"Xy_val.npz")

steps_per_epoch, n_events = get_n_iterations(fname_train, batch_size=BATCH_SIZE)
print(steps_per_epoch, n_events)

#validation_steps, n_evts_val = get_n_iterations(fnames_val[:N_FILES], batch_size=BATCH_SIZE)
#print(validation_steps, n_evts_val)

prediction_steps, n_evts_test = get_n_iterations(fname_test, batch_size=BATCH_SIZE)
print(prediction_steps, n_evts_test)
validation_steps, n_evts_val = get_n_iterations(fname_test, batch_size=BATCH_SIZE)
print(validation_steps, n_evts_val)

training_generator = data_generator(fname_train, batch_size=BATCH_SIZE,
                                    ftarget=lambda y: y)

validation_generator = data_generator(fname_val, batch_size=BATCH_SIZE,
                                      fdata=get_Time_Coord, ftarget=process_cosz)
#data_dir = DATA_DIR_IH

model = get_unet()
model.summary()

training_history = train_neural_network(model, training_generator, steps_per_epoch,
                                        validation_generator,
                                        validation_steps, batch_size=BATCH_SIZE)


model.fit(x=X_train, y=y_train, epochs=200, batch_size=2, verbose=1, validation_split=.2)

model.save(os.path.join(data_dir,"trained_UNet_200epochs.hdf5"))

print('Saving Model (JSON), Training History & Weights...', end='')
model_json_str = model.to_json()
with open(MODEL_JSON_FILEPATH, 'w') as model_json_f:
    model_json_f.write(model_json_str)

history_filepath = HISTORY_FILEPATH
dump(training_history.history, open(history_filepath, 'wb'))

model.save_weights(TRAINING_WEIGHTS_FILEPATH)
print('...Done!')
