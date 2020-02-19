import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
try:
    from .utils.data import define_dataset, select_dist
    from .utils.training import retrain_rUNet
except ModuleNotFoundError:
    from utils.data import define_dataset, select_dist
    from utils.training import retrain_rUNet
from models import rUNet, dice_loss

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"

DATA_DIR = DATA_DIR_DEEPTHOUGHT
SRC_DIR = os.path.join("/", "storage", "yw18581", "src", "leaf_reco")
ROOT_DIR = os.path.join(DATA_DIR, "dataset")
EXCLUDED = select_dist(dist_list=[1, 3, 15, 30], root_folder=ROOT_DIR)
print(EXCLUDED)

print("Load dataset")
data_loaders, data_lengths = define_dataset(root_folder=ROOT_DIR, batch_size=16, add_noise=10000,
                                            excluded_list=EXCLUDED, multi_processing=4)

print(data_lengths)
print("Define model")
coeffs = [0.40, 0.70]

n_epochs = 25

for coef in coeffs:
    print("combined loss: {}*dice_loss + {} mse".format(coef, 1.0 - coef))
    torch.cuda.empty_cache()
    print("Train model")
    model = rUNet(out_size=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_file = os.path.join(SRC_DIR, 'saved_models', 'trained_6positions_multi_loss',
                                   'Trained_rUNet_pytorch_6positions_dataset_100epochs_{}coeff_mask.pkl'.format(coef))

    history = retrain_rUNet(model=model, optimizer=optimizer,
                            criterion_dist=nn.MSELoss(), criterion_mask=dice_loss,
                            loss_coeff=coef, data_loaders=data_loaders,
                            data_lengths=data_lengths, checkpoint_file=checkpoint_file,
                            epochs=n_epochs, batch_size=16,
                            model_checkpoint=10, src_dir='/storage/yw18581/src/leaf_reco',
                            task_folder_name="trained_6positions_multi_loss_noise",
                            dataset_key="6positions", writer=True)
    print("Done")
