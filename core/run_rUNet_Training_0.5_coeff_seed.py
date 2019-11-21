import os
from utils.data import define_dataset, select_dist
from utils.training import training_phase_rUNet_multi_loss
import numpy as np
from models import rUNet, dice_loss
import torch
import torch.optim as optim
import torch.nn as nn

SEED = 8
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"
data_dir = DATA_DIR_DEEPTHOUGHT
src_dir = os.path.join("/", "storage", "yw18581", "src", "leaf_reco")
root_dir = os.path.join(data_dir, "dataset")
selected_distances = select_dist(dist_list=[2, 4, 10, 20, 25, 35], root_folder=root_dir)
print(selected_distances)

print("Load dataset")
data_loaders, data_lengths = define_dataset(root_folder=root_dir, batch_size=16, include_list=selected_distances,
                                            multi_processing=4)

print(data_lengths)
print("Define model")
coeffs = [0.50]
#coeffs_2 = [0.25, 0.30, 0.40]
n_epochs = 100

for coef in coeffs:
    print("combined loss: {}*dice_loss + {} mse".format(coef, 1.0 - coef))
    torch.cuda.empty_cache()
    print("Train model")
    model = rUNet(out_size=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = training_phase_rUNet_multi_loss(model=model, optimizer=optimizer, loss_coeff=coef,
                                   criterion_dist=nn.MSELoss(), criterion_mask=dice_loss,
                                   src_dir='/storage/yw18581/src/leaf_reco',
                                   task_folder_name="trained_6positions_0.50_multi_loss",
                                   data_loaders=data_loaders, data_lengths=data_lengths,
                                   epochs=n_epochs, batch_size=32, model_checkpoint=5,
                                   dataset_key="6positions", writer=True)
    print("Done")
