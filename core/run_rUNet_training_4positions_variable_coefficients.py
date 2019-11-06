import os
from .utils.data import define_dataset, select_dist
from .utils.training import training_phase_rUNet

from .models import rUNet, dice_loss
import torch
import torch.optim as optim
from torch import nn
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"
data_dir = DATA_DIR_DEEPTHOUGHT
src_dir = os.path.join("/", "storage", "yw18581", "src", "leaf_reco")
root_dir = os.path.join(data_dir, "dataset")
excluded = select_dist(dist_list=[1, 3, 15, 20, 30, 35], root_folder=root_dir)
print(excluded)

print("Load dataset")
data_loaders, data_lengths = define_dataset(root_folder=root_dir, batch_size=16, excluded_list=excluded,
                                            multi_processing=4)

print(data_lengths)
print("Define model")
coeffs = [0.75, 0.70, 0.60, 0.50]

n_epochs = 30

for coef in coeffs:
    print("combined loss: {}*dice_loss + {} mse".format(coef, 1.0 - coef))
    torch.cuda.empty_cache()
    print("Train model")
    model = rUNet(out_size=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = training_phase_rUNet(model=model, optimizer=optimizer, loss_coeff=coef,
                                   criterion_mask=dice_loss, criterion_dist=nn.MSELoss(),
                                   src_dir='/storage/yw18581/src/leaf_reco',
                                   task_folder_name="trained_4positions",
                                   data_loaders=data_loaders, data_lengths=data_lengths,
                                   epochs=n_epochs, batch_size=16, model_checkpoint=5,
                                   dataset_key="4positions", writer=True)
    print("Done")
