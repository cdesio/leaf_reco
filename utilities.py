import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm, trange
from Transformers import ChannelsFirst, ToTensor, Rescale, Cut, splitter_train_val_test, splitter
from DataSets import UNetDatasetFromFolders
import torch.optim as optim
from cUNet_pytorch_pooling import cUNet, dice_loss

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


def combined_loss(pred_mask, true_mask, pred_dist, true_dist, coeff):
    criterion_mask = dice_loss
    criterion_dist = nn.MSELoss()
    loss_mask = criterion_mask(pred_mask, true_mask)
    loss_dist = criterion_dist(pred_dist, true_dist)
    loss = coeff * loss_mask + (1.0-coeff) * loss_dist
    return loss
def create_history():
    history = {}
    history.setdefault("train", [])
    history.setdefault("val", [])
    history.setdefault("epochs", [])
    return history

def training_phase_rUNet(model, optimizer, loss_coeff,
                         data_loaders, data_lengths, epochs, batch_size, model_checkpoint, dev=0,
                         dataset_key="complete",
                         model_prefix="Trained_rUNet_pytorch_{}_dataset".format(dataset_key),
                         writer = None):

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = create_history()

    for epoch in trange(epochs, desc = "Training Epochs"):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0

            for i, batch in tqdm(enumerate(data_loaders[phase]), total = data_lengths[phase]//batch_size, desc="Mini Batch"):
                inputs = batch['image'.float().to(device)]
                labels_mask = batch['mask'].float().to(device)
                labels_dist = batch['dist'][..., np.newaxis].float().to(device)
                optimizer.zero_grad()
                out_mask, out_dist = model(inputs)
                loss = combined_loss(out_mask, labels_mask, out_dist, labels_dist, coeff=loss_coeff)

                if phase== "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            history[phase].append(running_loss)
        history['epochs'].append(epoch)

        if epoch%model_checkpoint==(model_checkpoint-1):
            torch.save(model.state_dict(), os.path.join("model", model_prefix+"_{}epochs_{}coeff_mask.pkl".format(epoch+1, loss_coeff )))
            epoch_loss = running_loss/data_lengths['phase']
            print('{} Loss: {:.4f)'.format(phase, epoch_loss))

        if writer:
            writer.add_scalar('Training_loss', running_loss, epoch)

    print("Finished training")
    print('Saving trained model')
    torch.save(model.state_dict(), os.path.join("model", model_prefix+"_{}epochs_{}coeff_mask_FINAL.pkl".format(epochs, loss_coeff )))

    return history



def inference_phase_rUNet(model, model_name, data_loaders, data_lengths, batch_size, dev=0):

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_name))
    model.eval()
    model.to(device);

    y_true = []
    y_pred = []

    for i, batch in tqdm(enumerate(data_loaders["test"]), total=data_lengths['test']//batch_size, desc = "Batch"):
        true_images, true_dists = batch['image'], batch['dist']
        _, pred_dists = model(true_images.float().to(device))
        for j, (img, tr_dist, pr_dist) in enumerate(zip(true_images,
                                                        true_dists.cpu().detach().numpy(),
                                                        pred_dists.cpu().detach().numpy())):
            y_true.append(tr_dist)
            y_pred.append(pr_dist)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred).ravel
        return y_true, y_pred


def inference_phase_rUNet_plot(model, model_name, data_loaders, data_lengths, batch_size, stop = 1, dev=0):
    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_name))
    model.eval()
    model.to(device);

    y_true = []
    y_pred = []

    for i, batch in tqdm(enumerate(data_loaders['test']), total=data_lengths['test'] // batch_size, desc='Batch'):

        true_images, true_masks, true_dists = batch['image'], batch['mask'], batch['dist']
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

