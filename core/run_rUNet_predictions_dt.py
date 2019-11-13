import os, sys
from sklearn.metrics import mean_squared_error
import re
import numpy as np
from utils.data import define_dataset, select_dist
from utils.inference import inference_phase_rUNet
from models import rUNet
import torch


#data_dir = sys.argv[1]
#saved_models = sys.argv[2]
data_dir = '/storage/yw18581/data'
saved_models = '/storage/yw18581/src/leaf_reco/saved_models/trained_6positions'

distances_unseen = [2, 4, 10, 20, 25, 35]
distances_testdata = [1, 3, 15, 30]

dataset_folder = os.path.join(data_dir, 'dataset')


regex = re.compile(r'\d+')


excluded_testdata = select_dist(dist_list=distances_testdata, root_folder=dataset_folder)
excluded_unseen = select_dist(dist_list=distances_unseen, root_folder=dataset_folder)

data_loaders_test, data_lengths_test = define_dataset(dataset_folder, batch_size=8,
                                                      excluded_list=excluded_testdata,
                                                      alldata=False)

print(data_lengths_test)

data_loaders_unseen, data_lengths_unseen = define_dataset(dataset_folder, batch_size=8,
                                                          excluded_list=excluded_unseen,
                                                          alldata=True)
print(data_lengths_unseen)


model_names = os.listdir(saved_models)

def get_fnames(key):
    f_list = []
    epochs = []
    for fname in model_names:
        if fname.startswith("Trained"):
            if regex.findall(fname.split("_")[6])[1]==str(key):
                f_list.append(fname)
                epochs.append(int(regex.findall(fname.split("_")[5])[0]))
    return f_list, epochs


def mse_vs_epochs(coeff, key, data_loaders, data_lengths, only_test=True):
    mse = []
    f_list, epochs = get_fnames(coeff)
    for fname, e in zip(f_list, epochs):
        print(fname, e)

        torch.cuda.empty_cache()
        model = rUNet(out_size=1)
        checkpoint = torch.load(os.path.join(saved_models, fname))['model_state_dict'];
        model.load_state_dict(checkpoint)
        y_true, y_pred = inference_phase_rUNet(model, data_loaders, data_lengths, batch_size=16,
                                               notebook=False, test=only_test)
        np.savez_compressed(os.path.join(saved_models, '_'.join(['predicted',
                                                                 key,
                                                                 fname.split('_')[3],
                                                                 fname.split('_')[5],
                                                                 fname.split('_')[6]])
                                         + '.npz'), true=y_true, pred=y_pred)
        mse.append(mean_squared_error(y_true, y_pred))

    #plt.plot(epochs, mse)
    return mse, epochs

#coeff_list_test = [25, 3, 4, 5, 6, 7, 75]

#coeff_list_test = sys.argv[3]
#for c in coeff_list_test:
#    _, _ = mse_vs_epochs(c, 'testdata', data_loaders_test, data_lengths_test)

coeff_list_unseen = [25, 3, 4]
#coeff_list_unseen = sys.argv[4]

for cu in coeff_list_unseen:
    _, _ = mse_vs_epochs(cu, 'unseen', data_loaders_unseen, data_lengths_unseen, only_test=False)
