import os, sys
from sklearn.metrics import mean_squared_error
import re
import numpy as np
from utils.data import define_dataset, select_dist
from utils.inference import inference_phase_rUNet
from models import rUNet
import torch
from functools import partial

#data_dir = sys.argv[1]
#saved_models = sys.argv[2]
data_dir = '/data/uob'
saved_models = '/data/uob/saved_models/trained_6positions_multi_loss'

training_dist = [2, 4, 10, 20, 25, 35]
unseen_dist = [1, 3, 15, 30]

dataset_folder = os.path.join(data_dir, 'dataset')
root_path = partial(os.path.join, saved_models)

regex = re.compile(r'\d+')


selected_testdata = select_dist(dist_list=training_dist, root_folder=dataset_folder)
selected_unseen = select_dist(dist_list=unseen_dist, root_folder=dataset_folder)

data_loaders_test, data_lengths_test = define_dataset(dataset_folder, batch_size=8,
                                                      include_list=selected_testdata,
                                                      alldata=False)

print(data_lengths_test)

data_loaders_unseen, data_lengths_unseen = define_dataset(dataset_folder, batch_size=8,
                                                          include_list=selected_unseen,
                                                          alldata=True)
print(data_lengths_unseen)


model_names = os.listdir(saved_models)

def get_fnames(coeff, prefix='Trained'):
    idx_coef = 6 if prefix=='Trained' else 4
    idx_epoch = 5 if prefix =='Trained' else 3
    f_list = []
    epochs = []
    for fname in model_names:
        if fname.startswith(prefix):
            if regex.findall(fname.split("_")[idx_coef])[1]==str(coeff):
                epoch = int(regex.findall(fname.split("_")[idx_epoch])[0]) 
                epochs.append(epoch)
                if int(regex.findall(fname.split("_")[idx_epoch])[0])==epoch:
                    f_list.append(fname)
    
    return np.array(f_list)[np.argsort(epochs)], np.sort(epochs)

def predict_dist(coeff, epoch,data_loaders, data_lengths, key, only_test=True):
    filelist, epochs = get_fnames(coeff)
    model_fname = filelist[int(np.argwhere(epochs==epoch)[0])]
    print(model_fname)
    
    torch.cuda.empty_cache()
    model = rUNet(out_size=1)
    
    checkpoint = torch.load(root_path(model_fname))['model_state_dict']
    model.load_state_dict(checkpoint)
    y_true, y_pred = inference_phase_rUNet(model, data_loaders, data_lengths, batch_size=8,
                                               notebook=False,test=only_test)
    print('saving to {}'.format('_'.join(['predicted',
                                                                 key,
                                                                 model_fname.split('_')[3],
                                                                 model_fname.split('_')[5],
                                                                 model_fname.split('_')[6]])
                                         + '.npz'))
    np.savez_compressed(os.path.join(saved_models, '_'.join(['predicted',
                                                                 key,
                                                                 model_fname.split('_')[3],
                                                                 model_fname.split('_')[5],
                                                                 model_fname.split('_')[6]])
                                         + '.npz'), true=y_true, pred=y_pred)
        
    mse=mean_squared_error(y_true, y_pred)
    print(mse)
    return

#coeff_list_test = [5, 7]
coeff_list_test = [25, 3, 4, 6]
#coeff_list_test = sys.argv[3]

for c in coeff_list_test:
    for e in range(5,105,5):
    	_ = predict_dist(c,e, data_loaders_test, data_lengths_test, 'testdata',only_test=True)

#coeff_list_unseen = [5,7]
#coeff_list_unseen = sys.argv[4]
coeff_list_unseen = [25, 3, 4, 6]
for cu in coeff_list_unseen:
    for e in range(5,105,5):
        _  = predict_dist(cu,e, data_loaders_unseen, data_lengths_unseen, 'unseen',only_test=False)

