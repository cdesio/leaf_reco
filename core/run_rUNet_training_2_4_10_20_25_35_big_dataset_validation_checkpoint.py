import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from tqdm.notebook import tqdm, trange
from .utils.data import ChannelsFirst, ToTensor, Rescale, Cut, splitter_train_val_test
from .utils.data import UNetDatasetFromFolders
import torch.optim as optim
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt

import torchvision

DATA_DIR_DEEPTHOUGHT = os.path.join("/",'storage','yw18581','data')
data_dir = DATA_DIR_DEEPTHOUGHT

root_folder = os.path.join(data_dir, "dataset")

excluded = ["10x10_1mm_first_clean", "10x10_3mm_first_clean", "10x10_15mm_first_clean", "10x10_30mm_first_clean",
            "10x10_1mm_second_clean", "10x10_3mm_second_clean", "10x10_15mm_second_clean", "10x10_30mm_second_clean",
            '10x10_1mm_third_clean', '10x10_3mm_third_clean', "10x10_15mm_third_clean", '10x10_30mm_third_clean']

composed = transforms.Compose([Cut(), Rescale(0.25), ChannelsFirst(), ToTensor()])

dataset = UNetDatasetFromFolders(root_folder, excluded=excluded, transform=composed)
print(len(dataset))

batch_size = 16
data_loaders, data_lengths = splitter_train_val_test(dataset,
                                                     validation_split=0.2,
                                                     test_split = 0.2,
                                                     batch=batch_size,
                                                     workers=4)
print(data_lengths['train'], data_lengths['val'], data_lengths['test'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from .models import rUNet, dice_loss

model = rUNet(out_size=1)

criterion_mask = dice_loss
criterion_dist = nn.MSELoss()

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('notebooks/runs/rUNet-6positions')

for i, batch in enumerate(data_loaders['test']):
    true_images, true_masks, true_dists = batch['image'], batch['mask'], batch['dist']
    img_grid = torchvision.utils.make_grid(true_images)
    mask_grid = torchvision.utils.make_grid(true_masks)
    #dist_grid = torchvision.utils.make_grid(true_dists)
    #matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image('true_images_grid', img_grid)
    writer.add_image('true_masks_grid', mask_grid)
    if i==2: 
        break

writer.add_graph(model, true_images.float().to(device))

writer.close()

epochs=50
coeff_mask = 0.75

# Training phase

for epoch in trange(epochs, desc = "Training Epochs"):

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)

        running_loss = 0.0
        for i, batch in tqdm(enumerate(data_loaders[phase]), total = data_lengths[phase]// batch_size, desc = "Mini Batch"):
            inputs = batch['image'].float().to(device)
            labels_mask = batch['mask'].float().to(device)
            labels_dist = batch['dist'][..., np.newaxis].float().to(device)

            optimizer.zero_grad()
            out_mask, out_dist = model(inputs)
            loss_mask = criterion_mask(out_mask, labels_mask)
            loss_dist = criterion_dist(out_dist, labels_dist)
            loss = coeff_mask * loss_mask + (1 - coeff_mask) * loss_dist

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
    if epoch%10==9:
        torch.save(model.state_dict(),
                           "model/TB_trained_rUnet_pytorch_regression_2_4_10_20_25_35_dataset_{}epochs_coeff_mask{}_validation.pkl".format(epoch+1,
                                                                                           coeff_mask) )
        epoch_loss = running_loss / data_lengths[phase]
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    
    writer.add_scalar('training_loss',
                         running_loss,
                         epoch)
print('Finished Training')



print('Saving trained model')
model_name = "model/TB__trained_rUnet_pytorch_regression_2_4_10_20_25_35_dataset_{}epochs_coeff_mask{}_validation.pkl".format(epochs, coeff_mask)

torch.save(model.state_dict(), model_name)

print('Inference step')

model_inference = rUNet(out_size=1)
model_inference.load_state_dict(torch.load(model_name))
model_inference.eval()
model_inference.to(device);

y_true = []
y_pred = []

for i, batch in tqdm(enumerate(data_loaders['test']), total=data_lengths['test']//batch_size, desc="Mini Batch"):
    true_images, true_dists = batch['image'], batch['dist']
    _, pred_dists = model_inference(true_images.float().to(device))
    print("batch {}".format(i + 1))
    for j, (img, tr_dist, pr_dist) in enumerate(zip(true_images,
                                                true_dists.cpu().detach().numpy(),
                                                pred_dists.cpu().detach().numpy())):
        true_dist = tr_dist
        pred_dist = pr_dist
        y_true.append(true_dist)
        y_pred.append(pred_dist)

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred).ravel()

print("mse: {}".format(mean_squared_error(y_true, y_pred)))




