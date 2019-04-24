import numpy as np
import matplotlib.pyplot as plt
from os import path as p
import os
from sklearn.metrics import mean_squared_error

data_dir = '/storage/yw18581/data/'
data_folder = os.path.join(data_dir, 'train_validation_test')

X_train = np.load(os.path.join(data_folder, 'Xy_train_dist.npz'))["y"]
y_train = np.load(os.path.join(data_folder, 'Xy_train_dist.npz'))["dist"]

X_val = np.load(os.path.join(data_folder,'Xy_val_dist.npz'))["y"]
y_val = np.load(os.path.join(data_folder, 'Xy_val_dist.npz'))["dist"]

X_test = np.load(os.path.join(data_folder, 'Xy_test_dist.npz'))["y"]
y_test = np.load(os.path.join(data_folder, 'Xy_test_dist.npz'))["dist"]

X_pred = np.load(os.path.join(data_dir, 'trained_models/UNet_training_generator_1500epochs/Xy_test_predicted_UNet.npz'))['y']
y_pred = np.load(os.path.join(data_dir, 'trained_models/UNet_training_generator_1500epochs/Xy_test_predicted_UNet.npz'))['dist']
print("data imported")
def cut_X(arr, reshape = None):
    x_cut = arr[:,960:1300,600:]
    if reshape:
        if len(x_cut.shape)>3:
            x_cut = x_cut[...,0]
            x_cut_out = x_cut.reshape(x_cut.shape[0],x_cut.shape[1]*x_cut.shape[2])
    else:
        x_cut_out = x_cut
    return x_cut_out

def reshape_RF(arr):
    arr_RF = arr.reshape((arr.shape[0], arr.shape[1]*arr.shape[2]))
    return arr_RF

X_train_cut = cut_X(X_train)

X_val_cut = cut_X(X_val)

X_test_cut = cut_X(X_test)

X_pred_cut = cut_X(X_pred)

X_train_RF = reshape_RF(X_train_cut)
X_val_RF = reshape_RF(X_val_cut)
X_test_RF = reshape_RF(X_test_cut)
N_EPOCHS = 30
BATCH_SIZE = 8

CHECKPOINT_FOLDER_PATH = p.join(data_dir, 'trained_models')
TASK_NAME = 'Leaf_position_regression'
TASK_FOLDER_PATH = p.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not p.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.externals import joblib
print("run RF")

def rf_training(depth, estimators,X,y):
    model = RandomForestRegressor(max_depth=depth, n_estimators=estimators, n_jobs=-1,
                                 criterion='mse')
    model.fit(X,y)
    score = model.score(X,y)
    print("score:{}, depth:{}, estimators:{}".format(score,depth,estimators))
    return model, score

estimators = np.arange(2,102,2)
depths = np.arange(1,11)

scores = []
for d in tqdm(depths[4:]):
    for est in tqdm(estimators):
        model, score = rf_training(d,est, X_train_RF, y_train)
        scores.append((score, d, est))
        filename = os.path.join(TASK_FOLDER_PATH,'leaf_regression_RF_{}est_{}depth.pkl'.format(est, d))
        print(filename.split('/')[-1])
        _ = joblib.dump(model, filename, compress=9)
        
        
print("done")