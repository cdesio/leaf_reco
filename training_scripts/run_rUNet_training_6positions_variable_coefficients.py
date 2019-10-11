import sys, os
sys.path.append('/storage/yw18581/src/leaf_reco')
from setenv import add_folders
add_folders()
from pickle import dump
from utility_functions import define_dataset, training_phase_rUNet, exclude_dist

from cUNet_pytorch_pooling import cUNet
import torch
import torch.optim as optim

DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"
data_dir = DATA_DIR_DEEPTHOUGHT
src_dir = os.path.join("/","storage", "yw18581", "src", "leaf_reco")
root_dir = os.path.join(data_dir, "dataset")
excluded = exclude_dist(dist_list = [1,2,3,4,10,15,20,30], root_folder=root_dir)
print(excluded)
task_folder = "trained_2_positions"

print("Load dataset")
data_loaders, data_lengths = define_dataset(root_folder=root_dir, batch_size=16, excluded_list = excluded)

print(data_lengths)
print("Define model")
model = cUNet(out_size=1)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

coeffs = [0.75, 0.70, 0.60, 0.50]
n_epochs = 50

for coef in coeffs:
    print("combined loss: {}*dice_loss + {} mse".format(coef, 1.0-coef))

    torch.cuda.empty_cache()
    print("Train model")
    history = training_phase_rUNet(optimizer=optimizer, loss_coeff=coef,
                               task_folder_name= task_folder,
                               data_loaders=data_loaders, data_lengths=data_lengths,
                               epochs=n_epochs, batch_size=16, model_checkpoint=10,
                               dataset_key="2positions", writer=True)
    history_filepath = os.path.join(src_dir, "saved_models", task_folder, "history_2positions_{}epochs_{}coef.pkl".format(n_epochs, coef))
    print("Save history to {}".format(history_filepath))
    dump(history, open(history_filepath, 'wb'))
    print("Done")

