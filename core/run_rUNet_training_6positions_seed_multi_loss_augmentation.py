import os
try:
    from .utils.data import define_dataset, select_dist
    from .utils.training import training_phase_rUNet_multi_loss
    from .models import rUNet, dice_loss
    from .utils.data.transformers import ChannelsFirst, Rescale, ToTensor, Crop, GaussianNoise, RandomCrop, Swap, \
        FlipUD, FlipLR
except ModuleNotFoundError:
    from utils.data import define_dataset, select_dist
    from utils.training import training_phase_rUNet_multi_loss
    from models import rUNet, dice_loss
    from utils.data.transformers import ChannelsFirst, Rescale, ToTensor, Crop, GaussianNoise, RandomCrop, Swap, \
        FlipUD, FlipLR

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"
data_dir = DATA_DIR_DEEPTHOUGHT
src_dir = os.path.join("/", "storage", "yw18581", "src", "leaf_reco")
root_dir = os.path.join(data_dir, "dataset")
selected_distances = select_dist(dist_list=[2, 4, 10, 20, 25, 35], root_folder=root_dir)
print(selected_distances)

base_transformers = [Crop(row_slice=slice(0,1400), col_slice=slice(1000,None)),
                           Rescale(scale=0.25), ChannelsFirst(), ToTensor()]

print("Define model")
#coeffs = [0.70, 0.75, 0.60]
coeffs = [0.40]
n_epochs = 100

for coef in coeffs:

    train_transformers = [RandomCrop(p=1), Swap(p=0.7), FlipLR(p=0.7), FlipUD(p=0.7),
                          GaussianNoise(p=0.75, mean=150, sigma=1), Rescale(0.25), ChannelsFirst(), ToTensor()]
    print("Load dataset")

    data_loaders, data_length = define_dataset(root_folder=ROOT_DIR, base_transformers=base_transformers,
                                               train_transformers=train_transformers,
                                               batch_size=16, include_list=selected_distances,
                                               alldata=False, multi_processing=4)
    print("combined loss: {}*dice_loss + {} mse".format(coef, 1.0 - coef))
    torch.cuda.empty_cache()
    print("Train model")
    model = rUNet(out_size=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = training_phase_rUNet_multi_loss(model=model, optimizer=optimizer, loss_coeff=coef,
                                   criterion_dist=nn.MSELoss(), criterion_mask=dice_loss,
                                   src_dir='/storage/yw18581/src/leaf_reco',
                                   task_folder_name="trained_fromscratch_6positions_multi_loss_augmentation_noise150",
                                   data_loaders=data_loaders, data_lengths=data_length,
                                   epochs=n_epochs, batch_size=16, model_checkpoint=5,
                                   dataset_key="6positions", writer=True)
    print("Done")
