#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import os
import os.path as p
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[2]:


data_dir = '/storage/yw18581/data/'
data_folder = os.path.join(data_dir, 'train_validation_test')


# In[3]:


X_train = np.load(os.path.join(data_folder, 'Xy_train_dist.npz'))["y"]
y_train = np.load(os.path.join(data_folder, 'Xy_train_dist.npz'))["dist"]


# In[4]:


X_val = np.load(os.path.join(data_folder,'Xy_val_dist.npz'))["y"]
y_val = np.load(os.path.join(data_folder, 'Xy_val_dist.npz'))["dist"]


# In[5]:


X_test = np.load(os.path.join(data_folder, 'Xy_test_dist.npz'))["y"]
y_test = np.load(os.path.join(data_folder, 'Xy_test_dist.npz'))["dist"]


# In[6]:


X_test_5classes = np.load(os.path.join(data_folder, "Xy_test_strat_dist_5classes.npz"))["y"]
y_test_5classes = np.load(os.path.join(data_folder, "Xy_test_strat_dist_5classes.npz"))["dist"]


# In[7]:


X_15mm = np.load(os.path.join(data_folder, "Xy_15mm.npz"))["y"]
y_15mm = np.load(os.path.join(data_folder, "Xy_15mm.npz"))["dist"]


# In[8]:


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


# In[9]:


X_train_cut = cut_X(X_train)
X_val_cut = cut_X(X_val)
X_test_cut = cut_X(X_test)
X_15mm_cut = cut_X(X_15mm)
X_train_RF = reshape_RF(X_train_cut)
X_val_RF = reshape_RF(X_val_cut)
X_test_RF = reshape_RF(X_test_cut)
X_15mm_RF  = reshape_RF(X_15mm_cut)


# In[10]:


CHECKPOINT_FOLDER_PATH = p.join(data_dir, 'trained_models')
TASK_NAME = 'Leaf_position_regression'
TASK_FOLDER_PATH = p.join(CHECKPOINT_FOLDER_PATH, TASK_NAME)

if not p.exists(TASK_FOLDER_PATH):
    os.makedirs(TASK_FOLDER_PATH)


# In[11]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


# estimators = np.arange(2,102,10)
# depths = np.arange(2,22,2)

# In[12]:


estimators = (2, 4, 8, 10, 20, 30, 50, 70, 100, 200, 300)


# In[13]:


tuned_parameteres = {"n_estimators": estimators}#, 'max_depth': depths}#,
                    #'min_samples_split': [1, 2, 3]}


# In[14]:


X_train_RF = np.vstack((X_train_RF, X_val_RF))
print(X_train_RF.shape)


# In[15]:


y_train = np.hstack((y_train,y_val))


# In[16]:


y_train.shape


# In[17]:


print(X_train_RF.shape, y_train.shape)


# In[18]:


from sklearn.model_selection import StratifiedKFold
X = X_train_RF
y = y_train
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)


# In[ ]:


model = GridSearchCV(RandomForestRegressor(criterion='mse',verbose=1), 
                     tuned_parameteres, cv=skf, verbose=10, n_jobs=4)


# In[ ]:


model.fit(X_train_RF, y_train)


# In[ ]:


print(model.best_params_)


# In[ ]:

joblib.dump(model.best_estimator_, 'rf_best_estimator.pkl', compress=1)




