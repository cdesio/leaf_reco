import os
import numpy as np
import torch
import torch.optim as optim


try:
    from .utils.data import define_dataset, select_dist
    from .utils.training import training_UNet
    from .models import UNet, dice_loss
except ModuleNotFoundError:
    from utils.data import define_dataset, select_dist
    from utils.training import training_UNet
    from models import UNet, dice_loss


SEED = 8
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"
data_dir = DATA_DIR_DEEPTHOUGHT
src_dir = os.path.join("/", "storage", "yw18581", "src", "leaf_reco")
root_dir = os.path.join(data_dir, "dataset")
selected_distances = select_dist(dist_list=[2, 4, 10, 25], root_folder=root_dir, keys_list=['first'])
excluded_distances = select_dist(root_folder=root_dir, keys_list=['second', 'third'])
print(selected_distances)

print("Load dataset")
data_loaders, data_lengths = define_dataset(root_folder=root_dir, batch_size=32, include_list=selected_distances,
                                            excluded_list=excluded_distances, multi_processing=4)

print(data_lengths)
print("Define model")
n_epochs = 100

torch.cuda.empty_cache()
print("Train model")
model = UNet
optimizer = optim.Adam(model.parameters(), lr=1e-4)

history = training_UNet(model=model, optimizer=optimizer, criterion_mask=dice_loss,
                               src_dir='/storage/yw18581/src/leaf_reco',
                               task_folder_name="trained_UNet_4positions_firstbatch",
                               data_loaders=data_loaders, data_lengths=data_lengths,
                               epochs=n_epochs, batch_size=32, model_checkpoint=5,
                               dataset_key="4positions_firstbatch", writer=True)
print("Done")
