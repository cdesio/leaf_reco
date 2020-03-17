import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
try:
    from .utils.data import define_dataset, select_dist
    from .utils.training import retrain_rUNet
    from .models import rUNet, dice_loss
    from .utils.data.transformers import ChannelsFirst, Rescale, ToTensor, Crop, GaussianNoise, RandomCrop, Swap, \
        FlipUD, FlipLR

except ModuleNotFoundError:
    from utils.data import define_dataset, select_dist
    from utils.training import retrain_rUNet
    from models import rUNet, dice_loss
    from utils.data.transformers import ChannelsFirst, Rescale, ToTensor, Crop, GaussianNoise, RandomCrop, Swap, \
        FlipUD, FlipLR
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"

DATA_DIR = DATA_DIR_DEEPTHOUGHT
SRC_DIR = os.path.join("/", "storage", "yw18581", "src", "leaf_reco")
ROOT_DIR = os.path.join(DATA_DIR, "dataset")
EXCLUDED = select_dist(dist_list=[1, 3, 15, 30], root_folder=ROOT_DIR)
print(EXCLUDED)

base_transformers = [Crop(row_slice=slice(0,1400), col_slice=slice(1000,None)),
                           Rescale(scale=0.25), ChannelsFirst(),ToTensor()]
train_transformers = [RandomCrop(p=1), Swap(p=0.7), FlipLR(p=0.7), FlipUD(p=0.7),
                      GaussianNoise(p=0.75, mean=30, sigma=2), Rescale(0.25), ChannelsFirst(),ToTensor()]
print("Load dataset")

data_loaders, data_length = define_dataset(root_folder=ROOT_DIR, base_transformers=base_transformers,
                                           train_transformers=train_transformers,
                                           batch_size=16, excluded_list=EXCLUDED,
                                            alldata=False, multi_processing=4)

print(data_length)
print("Define model")
coeffs = [0.40]

n_epochs = 50

for coef in coeffs:
    for noise in [10,25, 50, 75, 100]:
        train_transformers = [RandomCrop(p=1), Swap(p=0.7), FlipLR(p=0.7), FlipUD(p=0.7),
                              GaussianNoise(p=0.75, mean=noise, sigma=.5), Rescale(0.25), ChannelsFirst(), ToTensor()]
        print("Load dataset")

        data_loaders, data_length = define_dataset(root_folder=ROOT_DIR, base_transformers=base_transformers,
                                                   train_transformers=train_transformers,
                                                   batch_size=16, excluded_list=EXCLUDED,
                                                   alldata=False, multi_processing=4)
        print("combined loss: {}*dice_loss + {} mse".format(coef, 1.0 - coef))
        torch.cuda.empty_cache()
        print("Train model")
        model = rUNet(out_size=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        checkpoint_file = os.path.join(SRC_DIR, 'saved_models', 'trained_6positions_multi_loss',
                                       'Trained_rUNet_pytorch_6positions_dataset_100epochs_{}coeff_mask_gaus{}.pkl'.format(coef, noise))
        print(torch.load(checkpoint_file).keys())
        history = retrain_rUNet(model=model, optimizer=optimizer,
                                criterion_dist=nn.MSELoss(), criterion_mask=dice_loss,
                                loss_coeff=coef, data_loaders=data_loaders,
                                data_lengths=data_length, checkpoint_file=checkpoint_file,
                                epochs=n_epochs, batch_size=16,
                                model_checkpoint=5, src_dir='/storage/yw18581/src/leaf_reco',
                                task_folder_name="trained_6positions_multi_loss_augmentation_gaus",
                                dataset_key="6positions", writer=True)
        print("Done")
