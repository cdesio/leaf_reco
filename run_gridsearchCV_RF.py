import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import os
import os.path as p
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data_dir = '/storage/yw18581/data/'
data_folder = os.path.join(data_dir, 'train_validation_test')

X_train = np.load(os.path.join(data_folder, 'Xy_train_dist.npz'))["y"]
y_train = np.load(os.path.join(data_folder, 'Xy_train_dist.npz'))["dist"]

X_val = np.load(os.path.join(data_folder,'Xy_val_dist.npz'))["y"]
y_val = np.load(os.path.join(data_folder, 'Xy_val_dist.npz'))["dist"]


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
X_train_RF = reshape_RF(X_train_cut)
X_val_RF = reshape_RF(X_val_cut)

CHECKPOINT_FOLDER_PATH = p.join(data_dir, 'trained_models')
TASK_NAME = 'Leaf_position_regression'
TASK_FOLDER_PATH = p.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not p.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

estimators = np.arange(2,102,2)
depths = np.arange(2,22,2)

tuned_parameteres = {"n_estimators": estimators, 'max_depth': depths}#,
                    #'min_samples_split': [1, 2, 3]}

X_train_RF = np.vstack((X_train_RF, X_val_RF))

y_train = np.hstack((y_train,y_val))


from sklearn.model_selection import StratifiedKFold
X = X_train_RF
y = y_train
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    X_tr, X_tst = X[train_index], X[test_index]
    y_tr, y_tst = y[train_index], y[test_index]

model = GridSearchCV(RandomForestRegressor(criterion='mse',verbose=1), 
                     tuned_parameteres, cv=skf, verbose=1)
model.fit(X_train_RF, y_train)

filename = os.path.join(TASK_FOLDER_PATH,'leaf_regression_RF_grid_search_CV.pkl')
_ = joblib.dump(model, filename, compress=9)