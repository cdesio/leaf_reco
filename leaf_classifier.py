IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

import os
from os import path as p
import tensorflow as tf

from network_models_km3 import leaf_classification, train_neural_network
from data_loaders_km3 import data_generator, get_n_iterations

DATA_DIR_IH="/data/uob"
DATA_DIR_DEEPTHOUGHT="/storage/yw18581/data"

data_folder = DATA_DIR_IH
TRAIN_VAL_TEST_DIR = os.path.join(data_folder,"train_validation_test")

N_FILES = 1
BATCH_SIZE=3
N_EPOCHS = 10

CHECKPOINT_FOLDER_PATH = p.join(data_folder, 'trained_models')
TASK_NAME = 'CNN_leaf_classifier_training_{}epochs'.format(N_EPOCHS)
TASK_FOLDER_PATH = os.path.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not os.path.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

tf.keras.backend.clear_session()

model = leaf_classification(num_classes=4, kernel_size=3, pooling_size=3)

model.summary()

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


training_generator = data_generator(fname_train, batch_size=BATCH_SIZE,
                                    fdata = lambda y: y,
                                    ftarget=lambda dist: dist)

validation_generator = data_generator(fname_val, batch_size=BATCH_SIZE,
                                      fdata=lambda y: y,
                                      ftarget=lambda dist: dist)

tf.keras.backend.clear_session()

model = TZ_updown_classification(num_classes=4, kernel_size=3, pooling_size=3)

model.summary()

training_history = train_neural_network(model, training_generator, steps_per_epoch,
                                        validation_generator,
                                        validation_steps, batch_size=BATCH_SIZE, epochs = N_EPOCHS)

print('Saving Model (JSON), Training History & Weights...', end='')
model_json_str = model.to_json()
with open(MODEL_JSON_FILEPATH, 'w') as model_json_f:
    model_json_f.write(model_json_str)

history_filepath = HISTORY_FILEPATH
dump(training_history.history, open(history_filepath, 'wb'))

model.save_weights(TRAINING_WEIGHTS_FILEPATH)
print('...Done!')
