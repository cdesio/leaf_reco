import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data_dir = '/storage/yw18581/data/'
data_folder = os.path.join(data_dir, 'train_validation_test')

X_1mm = np.load(os.path.join(data_folder, "Xy_1mm.npz"))["y"]
y_1mm = np.load(os.path.join(data_folder, "Xy_1mm.npz"))["dist"]

X_2mm = np.load(os.path.join(data_folder, "Xy_2mm.npz"))["y"]
y_2mm = np.load(os.path.join(data_folder, "Xy_2mm.npz"))["dist"]

X_3mm = np.load(os.path.join(data_folder, "Xy_3mm.npz"))["y"]
y_3mm = np.load(os.path.join(data_folder, "Xy_3mm.npz"))["dist"]

X_4mm = np.load(os.path.join(data_folder, "Xy_4mm.npz"))["y"]
y_4mm = np.load(os.path.join(data_folder, "Xy_4mm.npz"))["dist"]


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
    

X_1mm_cut = cut_X(X_1mm)
X_2mm_cut = cut_X(X_2mm)
X_3mm_cut = cut_X(X_3mm)
X_4mm_cut = cut_X(X_4mm)

X_1mm_RF  = reshape_RF(X_1mm_cut)
X_2mm_RF  = reshape_RF(X_2mm_cut)
X_3mm_RF  = reshape_RF(X_3mm_cut)
X_4mm_RF  = reshape_RF(X_4mm_cut)

X = np.vstack((X_1mm_RF, X_2mm_RF, X_4mm_RF))
y = np.hstack((y_1mm, y_2mm, y_4mm))

estimations = np.arange(2,102,2)
print(estimations)

depths = np.arange(10,210,10)
print(depths)

d = []
for est in estimations:
    for dep in depths:
        rf = RandomForestRegressor(n_estimators=est, max_depth=dep, 
                                   random_state=42, verbose=0, n_jobs=2)
        rf.fit(X,y)
        print("fit for n_est:{}, depth:{}".format(est, dep))
        preds = rf.predict(X_3mm_RF)
        err = mean_squared_error(y_3mm, preds)
        d.append([rf, est, dep, err])
        
        
np.savez_compressed("/storage/yw18581/data/test_rf_1234.npz", db=d)

