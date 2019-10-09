import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
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

X_1mm = np.load(os.path.join(data_folder, "Xy_1mm.npz"))["y"]
y_1mm = np.load(os.path.join(data_folder, "Xy_1mm.npz"))["dist"]

X_3mm = np.load(os.path.join(data_folder, "Xy_3mm.npz"))["y"]
y_3mm = np.load(os.path.join(data_folder, "Xy_3mm.npz"))["dist"]

X_15mm = np.load(os.path.join(data_folder, "Xy_15mm.npz"))["y"]
y_15mm = np.load(os.path.join(data_folder, "Xy_15mm.npz"))["dist"]

X_20mm = np.load(os.path.join(data_folder, "Xy_20mm.npz"))["y"]
y_20mm = np.load(os.path.join(data_folder, "Xy_20mm.npz"))["dist"]

X_30mm = np.load(os.path.join(data_folder, "Xy_30mm.npz"))["y"]
y_30mm = np.load(os.path.join(data_folder, "Xy_30mm.npz"))["dist"]

X_35mm = np.load(os.path.join(data_folder, "Xy_35mm.npz"))["y"]
y_35mm = np.load(os.path.join(data_folder, "Xy_35mm.npz"))["dist"]

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

X_train_RF = reshape_RF(X_train_cut)
X_val_RF = reshape_RF(X_val_cut)
X_test_RF = reshape_RF(X_test_cut)

X_1mm_cut = cut_X(X_1mm)
X_3mm_cut = cut_X(X_3mm)
X_15mm_cut = cut_X(X_15mm)
X_20mm_cut = cut_X(X_20mm)
X_30mm_cut = cut_X(X_30mm)
X_35mm_cut = cut_X(X_35mm)

X_1mm_RF  = reshape_RF(X_1mm_cut)
X_3mm_RF  = reshape_RF(X_3mm_cut)
X_15mm_RF  = reshape_RF(X_15mm_cut)
X_20mm_RF  = reshape_RF(X_20mm_cut)
X_30mm_RF  = reshape_RF(X_30mm_cut)
X_35mm_RF  = reshape_RF(X_35mm_cut)

X = np.vstack((X_train_RF,X_val_RF))
y = np.hstack((y_train, y_val))


def predict_on_unknown_dist(model):
    mse_test = mean_squared_error(y_test, model.predict(X_test_RF))
    mse_1mm = mean_squared_error(y_1mm, model.predict(X_1mm_RF))
    mse_3mm = mean_squared_error(y_3mm, model.predict(X_3mm_RF))
    mse_15mm = mean_squared_error(y_15mm, model.predict(X_15mm_RF))
    mse_20mm = mean_squared_error(y_20mm, model.predict(X_20mm_RF))
    mse_30mm = mean_squared_error(y_30mm, model.predict(X_30mm_RF))
    mse_35mm = mean_squared_error(y_35mm, model.predict(X_35mm_RF))
    return mse_test, mse_1mm, mse_3mm, mse_15mm, mse_20mm, mse_30mm, mse_35mm


estimations = np.arange(2,42,2)
print(estimations)


depths = np.arange(10,210,10)
print(depths)



import pandas as pd

d = []
for est in estimations:
    for dep in depths:
        rf = RandomForestRegressor(n_estimators=est, max_depth=dep, 
                                   random_state=42, verbose=0, n_jobs=2)
        rf.fit(X,y)
        print("fit for n_est:{}, depth:{}".format(est, dep))
        test, p1,p3,p15,p20, p30, p35 = predict_on_unknown_dist(rf)
        d.append([est, dep, test, p1, p3, p15, p20, p30, p35])

df = pd.DataFrame(d, columns=["estimators", "depth","test", 
                      "1mm", "3mm", "15mm", "20mm",
                     "30mm", "35mm"])


np.savez_compressed(os.path.join(data_dir, "rf_results_dataframe.npz"), 
                                DF=df)
