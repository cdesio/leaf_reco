import torch
import numpy as np
from torchvision import transforms
from Transformers import ChannelsFirst, ToTensor, Rescale, Cut
from DataSets import UNetDatasetFromFolders
from cUNet_pytorch_pooling import cUNet, dice_loss
from torch.utils.data import DataLoader
import re, os
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

def define_dataset(root_folder, batch_size=16, validation_split = 0.2, test_split=0.2, excluded_list=None, scale=0.25):
    excluded = excluded_list
    composed = transforms.Compose([Cut(), Rescale(scale), ChannelsFirst(), ToTensor()])
    dataset = UNetDatasetFromFolders(root_folder, excluded = excluded, transform=composed)
    data_loaders, data_lengths = splitter_train_val_test(dataset,
                                                         validation_split,
                                                         test_split,
                                                         batch=batch_size,
                                                         workers = 4)
    return data_loaders, data_lengths


#def combined_loss(pred_mask, true_mask, pred_dist, true_dist, coeff):
#    criterion_mask = dice_loss
#    criterion_dist = nn.MSELoss()
#    loss_mask = criterion_mask(pred_mask, true_mask)
#    loss_dist = criterion_dist(pred_dist, true_dist)
#    loss = coeff * loss_mask + (1.0-coeff) * loss_dist
#    return loss

def create_history():
    history = {}
    history.setdefault("train", [])
    history.setdefault("val", [])
    history.setdefault("epochs", [])
    return history

def training_phase_rUNet(model, optimizer, loss_coeff,
                         data_loaders, data_lengths, epochs, batch_size, model_checkpoint,task_folder_name, dev=0,
                         dataset_key="complete",
                         model_prefix="Trained_rUNet_pytorch",
                         writer = None, notebook=None):
    src_dir = os.path.join("/","storage", "yw18581", "src", "leaf_reco")
    task_folder_path = os.path.join(src_dir,"saved_models",task_folder_name)
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange

    if writer:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(src_dir, 'notebooks','runs', 'rUNet-{}_dataset_{}epochs_{}coeff_mask.pkl'.format(dataset_key, epochs, loss_coeff)))


    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")

    model.to(device)
    history = create_history()
    criterion_mask = dice_loss
    criterion_dist = nn.MSELoss()

    for epoch in trange(epochs, desc = "Training Epoch"):
        print(epoch +1)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0

            for i, batch in tqdm(enumerate(data_loaders[phase]), total = data_lengths[phase]//batch_size, desc="Mini Batch {}".format(phase)):
                inputs = batch['image'].float().to(device)
                labels_mask = batch['mask'].float().to(device)
                labels_dist = batch['dist'][..., np.newaxis].float().to(device)
                optimizer.zero_grad()
                out_mask, out_dist = model(inputs)
                loss_mask = criterion_mask(out_mask, labels_mask)
                loss_dist = criterion_dist(out_dist, labels_dist)
                loss = (loss_coeff * loss_mask) + (1.0 - loss_coeff) * loss_dist

                if phase== "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                print(running_loss)
            epoch_loss = running_loss / (data_lengths[phase]//batch_size)

            print('{} Loss: {:.4f})'.format(phase, epoch_loss))
            if writer:

                if phase == 'train':
                    tb_writer.add_scalar('Training_loss', epoch_loss, epoch)
                else:
                    tb_writer.add_scalar('Validation_loss', epoch_loss, epoch)

            history[phase].append(epoch_loss)
        history['epochs'].append(epoch+1)

        if epoch%model_checkpoint==(model_checkpoint-1):
            torch.save(model.state_dict(), os.path.join(task_folder_path,  model_prefix+"_{}_dataset_{}epochs_{}coeff_mask.pkl".format(dataset_key, epoch+1, loss_coeff )))



    print("Finished training")
    print('Saving trained model')
    torch.save(model.state_dict(), os.path.join(task_folder_path, model_prefix+"_{}_dataset_{}epochs_{}coeff_mask_FINAL.pkl".format(dataset_key, epochs, loss_coeff )))

    return history



def inference_phase_rUNet(model, data_loaders, data_lengths, batch_size, dev=0, notebook=None):
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")

    #model = cUNet(out_size=1)

    #model.load_state_dict(torch.load(model_name))
    model.eval()
    model.to(device);

    y_true = []
    y_pred = []

    for i, batch in tqdm(enumerate(data_loaders["test"]), total=data_lengths["test"]//batch_size, desc = "Batch"):
        true_images, true_dists = batch["image"], batch["dist"]
        _, pred_dists = model(true_images.float().to(device))
        for j, (img, tr_dist, pr_dist) in enumerate(zip(true_images,
                                                        true_dists.cpu().detach().numpy(),
                                                        pred_dists.cpu().detach().numpy())):
            y_true.append(tr_dist)
            y_pred.append(pr_dist)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).ravel()
    return y_true, y_pred


def inference_phase_rUNet_plot_notebook(model, data_loaders, data_lengths, batch_size, stop = 1, dev=0):
    from tqdm.notebook import tqdm
    import matplotlib.pyplot as plt

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")
    #model = cUNet(out_size=1)

    #model.load_state_dict(torch.load(model_name))
    model.eval()
    model.to(device);

    for i, batch in tqdm(enumerate(data_loaders["test"]), total=data_lengths["test"] // batch_size, desc="Batch"):

        true_images, true_masks, true_dists = batch["image"], batch["mask"], batch["dist"]
        pred_masks, pred_dists = model(true_images.float().to(device))
        print("batch {}".format(i + 1))
        for j, (img, tr_msk, tr_dist, pr_msk, pr_dist) in enumerate(zip(true_images,
                                                                        true_masks,
                                                                        true_dists.cpu().detach().numpy(),
                                                                        pred_masks.cpu().detach().numpy(),
                                                                        pred_dists.cpu().detach().numpy())):

            print("{}: true_dist: {}, pred_dist: {}".format(j + 1, tr_dist, pr_dist))

            f = plt.figure(figsize=(10, 5))
            f.add_subplot(1, 3, 1)
            plt.imshow(img[0, ...], cmap='gray')
            f.add_subplot(1, 3, 2)
            plt.imshow(tr_msk[0, ...], cmap='gray')
            f.add_subplot(1, 3, 3)
            plt.imshow(pr_msk[0, ...], cmap='gray')
            plt.show(block=True)

        if i == stop:
            break
        return

def splitter(dataset, validation_split=0.2, batch=16, workers=4):

    dataset_len = len(dataset)
    indices = list(range(dataset_len))
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch, num_workers=workers)
    validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch, num_workers=workers)

    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(train_idx), "val": val_len}
    return data_loaders, data_lengths

def splitter_train_val_test(dataset, validation_split=0.2, test_split=0.2, batch=16, workers=4):

    dataset_len = len(dataset)
    indices = list(range(dataset_len))

    test_len = int(np.floor(test_split * dataset_len))
    train_len = dataset_len - test_len

    test_idx = np.random.choice(indices, size=test_len, replace=False)
    train_idx = list(set(indices) - set(test_idx))

    test_sampler = SubsetRandomSampler(test_idx)

    validation_len = int(np.floor(validation_split * train_len))
    validation_idx = np.random.choice(train_idx, size=validation_len, replace=False)
    train_idx_out = list(set(train_idx)- set(validation_idx))

    validation_sampler = SubsetRandomSampler(validation_idx)
    train_sampler = SubsetRandomSampler(train_idx_out)

    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch, num_workers=workers)
    validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=batch, num_workers=workers)
    test_loader = DataLoader(dataset, sampler = test_sampler, batch_size=batch, num_workers=workers)
    data_loaders = {"train": train_loader, "val": validation_loader, "test": test_loader}
    data_lengths = {"train": len(train_idx_out), "val": validation_len, "test": test_len}
    return data_loaders, data_lengths


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def exclude_dist(dist_list, root_folder):

    regex = re.compile(r'\d+')

    excluded = []
    for dist in dist_list:
        for root, dirs, _ in os.walk(root_folder):
            for d in dirs:
                if hasNumbers(d):
                    if int(regex.findall(d)[-1]) == dist:
                        excluded.append(d)
    return excluded