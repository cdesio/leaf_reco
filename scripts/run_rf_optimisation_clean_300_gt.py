


import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle



data_dir = '/storage/yw18581/data/'
data_folder = os.path.join(data_dir, 'train_validation_test')
clean_dir = os.path.join(data_folder, 'clean_300_june')


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

def cut_reshape(arr):
    arr_cut = cut_X(arr)
    arr_RF = reshape_RF(arr_cut)
    return arr_RF


#estimators = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 50]
#estimators = [50, 60, 70 , 72, 76, 80, 88, 96, 100, 120, 160, 200, 240, 300, 400, 500, 750, 960, 1000, 2000, 3000]
estimators = [3000, 4000, 5000, 8000, 10000, 12000]
Xy_train = np.load("/storage/yw18581/data/train_validation_test/Xy_train_clean_300_24_10_25.npz")


Xy_val = np.load("/storage/yw18581/data/train_validation_test/Xy_val_clean_300_24_10_25.npz")


Xy_test = np.load("/storage/yw18581/data/train_validation_test/Xy_test_clean_300_24_10_25.npz")


X_test = cut_reshape(Xy_test["y"])
y_test = Xy_test["dist"]


X_train = cut_reshape(Xy_train["y"])
y_train = Xy_train["dist"]
X_val = cut_reshape(Xy_val["y"])
y_val = Xy_val["dist"]


X_train = np.vstack((X_train, X_val))
y_train = np.hstack((y_train, y_val))


def import_no_split(pos, keyword):
    Xy = np.load(os.path.join(clean_dir,"Xy_"+pos+"_{}.npz".format(keyword)))
    X = Xy["y"]
    y = Xy["dist"]
    X_RF = cut_reshape(X)
    return X_RF, y

X_15_1, y_15_1 = import_no_split("15mm", "clean300_june")
X_15_2, y_15_2 = import_no_split("15mm", "second_batch_clean300_june")

X_15_gt = np.vstack((X_15_1, X_15_2))
y_15_gt = np.hstack((y_15_1, y_15_2))



errors_gt = []


for est in estimators:
    
    rf = RandomForestRegressor(random_state=42, n_estimators=est,
                                   n_jobs=2, verbose=2)
    rf.fit(X_train, y_train)
    preds_test_gt = rf.predict(X_test)
    preds_test_15mm_gt = rf.predict(X_15_gt)
        
        
    mse_gt = mean_squared_error(preds_test_gt, y_test)
        
    mse_15_gt = mean_squared_error(preds_test_15mm_gt, y_15_gt)
        
    errors_gt.append((rf, est, mse_gt, mse_15_gt))
    


pickle.dump(errors_gt, 
            open(os.path.join(data_dir, "trained_models", "RF_OPTIMISATION_CLEAN_300_GT_larger_est.npz"), 'wb'))
