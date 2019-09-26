import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import tqdm
from Transformers import ChannelsFirst, ToTensor, Rescale, Cut, splitter
from DataSets import UNetDataSetFromNpz
import torch.optim as optim
from cUNet_pytorch_pooling import cUNet, dice_loss
from sklearn.metrics import mean_squared_error

DATA_DIR_DEEPTHOUGHT = os.path.join("/",'storage','yw18581','data')
data_dir = DATA_DIR_DEEPTHOUGHT
train_test = os.path.join(data_dir, "train_validation_test")

data = np.load(os.path.join(train_test,"Xy_train+val_clean_300_24_10_25.npz"))
x = data["x"]
y = data['y']
dist = data['dist']

composed = transforms.Compose([Rescale(0.25), ChannelsFirst(), ToTensor()])

dataset_train = UNetDataSetFromNpz(x, y, transform=composed_npz, dist = dist[...,np.newaxis])
print(len(dataset_train))

train_loaders, train_lengths = splitter(dataset_train, validation_split=0.2, batch=batch_size, workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = cUNet(out_size=1)

criterion_mask = dice_loss
criterion_dist = nn.MSELoss()

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

batch_size=32
epochs=50
coeff_mask = 0.75

#Training phase

for epoch in tqdm.tqdm(range(epochs)):
    print("Epoch {}/{}\n".format(epoch + 1, epochs))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)

        running_loss = 0.0
        for i, batch in enumerate(train_loaders[phase]):
            inputs = batch['image'].float().to(device)
            labels_mask = batch['mask'].float().to(device)
            labels_dist = batch['dist'].float().to(device)

            optimizer.zero_grad()
            out_mask, out_dist = model(inputs)
            loss_mask = criterion_mask(out_mask, labels_mask)
            loss_dist = criterion_dist(out_dist, labels_dist)
            loss = coeff_mask * loss_mask + (1 - coeff_mask) * loss_dist
            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / train_lengths[phase]
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
print('Finished Training')

print('Saving trained model')
model_name = "../model/trained_cUNet_pytorch_regression_validation_{}epochs_coeff_mask{}_batch{}_on_npz_notranspose.pkl".format(epochs, coeff_mask, batch_size)

torch.save(model.state_dict(), model_name)

print('Inference step')

model_inference = cUNet(out_size=1)
model_inference.load_state_dict(torch.load(model_name))
model_inference = model.eval()
model_inference.to(device)

test_data = np.load(os.path.join(train_test,"Xy_test_clean_300_24_10_25.npz"))
x_test = test_data["x"]
y_test = test_data['y']
dist_test = test_data['dist']

test_dataset = UNetDataSetFromNpz(x_test, y_test, transform=composed_npz, dist = dist_test[...,np.newaxis])

test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


y_true = []
y_pred = []

for i, batch in enumerate(test_data_loader):
    true_images, true_dists = batch['image'], batch['dist']
    _, pred_classes = model_inference(true_images.float().to(device))
    print("batch {}".format(i + 1))
    for j, (img, tr_cl, pr_cl) in enumerate(zip(true_images,
                                                true_dists.cpu().detach().numpy(),
                                                pred_classes.cpu().detach().numpy())):
        true_dist = tr_cl
        pred_dist = pr_cl
        y_test.append(true_dist)
        y_pred.append(pred_dist)

y_pred = np.asarray(y_pred).ravel()
y_true = np.asarray(y_true)

print("mse: {}".format(mean_squared_error(y_true, y_pred)))