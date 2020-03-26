import torch
import numpy as np
import os
import torch.nn as nn
from pickle import dump


# def combined_loss(pred_mask, true_mask, pred_dist, true_dist, coeff):
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

def create_history_multi_loss():
    history = {}
    history.setdefault("train", [])
    history.setdefault("val", [])
    history.setdefault("train_dice", [])
    history.setdefault("train_mse", [])
    history.setdefault("val_dice", [])
    history.setdefault("val_mse", [])
    history.setdefault("epochs", [])
    return history


def training_phase_rUNet(model, optimizer, criterion_mask, criterion_dist, loss_coeff, src_dir,
                         data_loaders, data_lengths, epochs, batch_size, model_checkpoint, task_folder_name, dev=0,
                         dataset_key="complete",
                         model_prefix="Trained_rUNet_pytorch",
                         writer=None, notebook=None):
    task_folder_path = os.path.join(src_dir, "saved_models", task_folder_name)
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange

    if writer:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(src_dir, 'notebooks', 'runs',
                                               'rUNet-{}_dataset_{}epochs_{}coeff_mask.pkl'.format(dataset_key, epochs,
                                                                                                   loss_coeff)))

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")

    model.to(device)
    history = create_history()

    for epoch in trange(epochs, desc="Training Epoch"):
        print(epoch + 1)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0

            for i, batch in tqdm(enumerate(data_loaders[phase]), total=data_lengths[phase] // batch_size,
                                 desc="Mini Batch {}".format(phase)):
                inputs = batch['image'].float().to(device)
                labels_mask = batch['mask'].float().to(device)
                labels_dist = batch['dist'][..., np.newaxis].float().to(device)
                optimizer.zero_grad()
                out_mask, out_dist = model(inputs)
                loss_mask = criterion_mask(out_mask, labels_mask)
                loss_dist = criterion_dist(out_dist, labels_dist)
                loss = (loss_coeff * loss_mask) + (1.0 - loss_coeff) * loss_dist

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                # print(running_loss)
            epoch_loss = running_loss / (data_lengths[phase] // batch_size)

            if phase == 'train':
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                train_loss = epoch_loss
            else:
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                val_loss = epoch_loss
            if writer:

                if phase == 'train':
                    tb_writer.add_scalar('Training_loss', epoch_loss, epoch)
                else:
                    tb_writer.add_scalar('Validation_loss', epoch_loss, epoch)

            history[phase].append(epoch_loss)
        history['epochs'].append(epoch)

        if epoch % model_checkpoint == (model_checkpoint - 1) or epoch == epochs - 1:
            model_filepath = os.path.join(task_folder_path,
                                          model_prefix + "_{}_dataset_{}epochs_{}coeff_mask.pkl".format(dataset_key,
                                                                                                        epoch + 1,
                                                                                                        loss_coeff))

            print("Save model checkpoint to: {}".format(model_filepath))

            torch.save(
                dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     train_loss=train_loss, val_loss=val_loss), model_filepath)

    history_filepath = os.path.join(task_folder_path,
                                    "history_" + model_prefix + "_{}epochs_{}coef.pkl".format(epochs, loss_coeff))
    print("Save history to {}".format(history_filepath))
    dump(history, open(history_filepath, 'wb'))

    print("Finished training")

    return history

def training_phase_rUNet_multi_loss(model, optimizer, criterion_mask, criterion_dist, loss_coeff, src_dir,
                         data_loaders, data_lengths, epochs, batch_size, model_checkpoint, task_folder_name, dev=0,
                         dataset_key="complete",
                         model_prefix="Trained_rUNet_pytorch",
                         writer=None, notebook=None):
    task_folder_path = os.path.join(src_dir, "saved_models", task_folder_name)
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange

    if writer:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(src_dir, 'notebooks', 'runs',
                                               'rUNet-{}_dataset_{}epochs_{}coeff_mask.pkl'.format(dataset_key, epochs,
                                                                                                   loss_coeff)))

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")

    model.to(device)
    history = create_history_multi_loss()

    for epoch in trange(epochs, desc="Training Epoch"):
        print(epoch + 1)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_dice_loss = 0.0
            running_mse_loss = 0.0

            for i, batch in tqdm(enumerate(data_loaders[phase]), total=data_lengths[phase] // batch_size,
                                 desc="Mini Batch {}".format(phase)):
                inputs = batch['image'].float().to(device)
                labels_mask = batch['mask'].float().to(device)
                labels_dist = batch['dist'][..., np.newaxis].float().to(device)
                optimizer.zero_grad()
                out_mask, out_dist = model(inputs)
                loss_mask = criterion_mask(out_mask, labels_mask)
                loss_dist = criterion_dist(out_dist, labels_dist)
                loss = (loss_coeff * loss_mask) + (1.0 - loss_coeff) * loss_dist

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_dice_loss += loss_mask.item()
                running_mse_loss += loss_dist.item()
                # print(running_loss)
            epoch_loss = running_loss / (data_lengths[phase] // batch_size)
            epoch_dice_loss = running_dice_loss / (data_lengths[phase] // batch_size)
            epoch_mse_loss = running_mse_loss / (data_lengths[phase] // batch_size)

            if phase == 'train':
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                train_loss = epoch_loss
            else:
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                val_loss = epoch_loss
            if writer:

                if phase == 'train':
                    tb_writer.add_scalar('Training_loss', epoch_loss, epoch)
                else:
                    tb_writer.add_scalar('Validation_loss', epoch_loss, epoch)

            history[phase].append(epoch_loss)
            history['{}_dice'.format(phase)].append(epoch_dice_loss)
            history['{}_mse'.format(phase)].append(epoch_mse_loss)
        history['epochs'].append(epoch)

        if epoch % model_checkpoint == (model_checkpoint - 1) or epoch == epochs - 1:
            model_filepath = os.path.join(task_folder_path,
                                          model_prefix + "_{}_dataset_{}epochs_{}coeff_mask.pkl".format(dataset_key,
                                                                                                        epoch + 1,
                                                                                                        loss_coeff))

            print("Save model checkpoint to: {}".format(model_filepath))

            torch.save(
                dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     train_loss=train_loss, val_loss=val_loss), model_filepath)

    history_filepath = os.path.join(task_folder_path,
                                    "history_" + model_prefix + "_{}epochs_{}coef.pkl".format(epochs, loss_coeff))
    print("Save history to {}".format(history_filepath))
    dump(history, open(history_filepath, 'wb'))

    print("Finished training")

    return history


def retrain_rUNet(model,
                  optimizer, criterion_mask, criterion_dist, loss_coeff,
                  data_loaders, data_lengths,
                  checkpoint_file, epochs, batch_size,
                  model_checkpoint, src_dir,
                  task_folder_name, dev=0, dataset_key="complete",
                  model_prefix="Trained_rUNet_pytorch",
                  writer=None, notebook=None):
    task_folder_path = os.path.join(src_dir, "saved_models", task_folder_name)
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange

    if writer:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(src_dir, 'notebooks', 'runs',
                                               'rUNet-{}_dataset_{}epochs_{}coeff_mask.pkl'.format(dataset_key, epochs,
                                                                                                   loss_coeff)))

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    model.to(device)
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    start_epoch = checkpoint['epoch']

    history = create_history()

    for epoch in trange(start_epoch + 1, start_epoch + 1 + epochs, desc="Training Epoch"):
        print(epoch + 1)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0

            for i, batch in tqdm(enumerate(data_loaders[phase]), total=data_lengths[phase] // batch_size,
                                 desc="Mini Batch {}".format(phase)):
                inputs = batch['image'].float().to(device)
                labels_mask = batch['mask'].float().to(device)
                labels_dist = batch['dist'][..., np.newaxis].float().to(device)
                optimizer.zero_grad()
                out_mask, out_dist = model(inputs)
                loss_mask = criterion_mask(out_mask, labels_mask)
                loss_dist = criterion_dist(out_dist, labels_dist)
                loss = (loss_coeff * loss_mask) + (1.0 - loss_coeff) * loss_dist

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                # print(running_loss)
            epoch_loss = running_loss / (data_lengths[phase] // batch_size)

            if phase == 'train':
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                train_loss = epoch_loss
            else:
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                val_loss = epoch_loss
            if writer:

                if phase == 'train':
                    tb_writer.add_scalar('Training_loss', epoch_loss, epoch)
                else:
                    tb_writer.add_scalar('Validation_loss', epoch_loss, epoch)

            history[phase].append(epoch_loss)
        history['epochs'].append(epoch)

        if epoch % model_checkpoint == (model_checkpoint - 1) or epoch == start_epoch + epochs - 1:
            model_filepath = os.path.join(task_folder_path,
                                          model_prefix + "_{}_dataset_{}epochs_{}coeff_mask.pkl".format(dataset_key,
                                                                                                        epoch + 1,
                                                                                                        loss_coeff))

            print("Save model checkpoint to: {}".format(model_filepath))

            torch.save(
                dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     train_loss=train_loss, val_loss=val_loss), model_filepath)

    history_filepath = os.path.join(task_folder_path,
                                    "history_" + model_prefix + "_{}epochs_{}coef.pkl".format(epochs, loss_coeff))
    print("Save history to {}".format(history_filepath))
    dump(history, open(history_filepath, 'wb'))

    print("Finished training")

    return history


def retrain_rUNet_multi_loss(model,
                  optimizer, criterion_mask, criterion_dist, loss_coeff,
                  data_loaders, data_lengths,
                  checkpoint_file, epochs, batch_size,
                  model_checkpoint, src_dir,
                  task_folder_name, dev=0, dataset_key="complete",
                  model_prefix="Trained_rUNet_pytorch",
                  writer=None, notebook=None):
    task_folder_path = os.path.join(src_dir, "saved_models", task_folder_name)
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange

    if writer:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(src_dir, 'notebooks', 'runs',
                                               'rUNet-{}_dataset_{}epochs_{}coeff_mask.pkl'.format(dataset_key, epochs,
                                                                                                   loss_coeff)))

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    model.to(device)
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    start_epoch = checkpoint['epoch']

    history = create_history_multi_loss()

    for epoch in trange(start_epoch + 1, start_epoch + 1 + epochs, desc="Training Epoch"):
        print(epoch + 1)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_dice_loss = 0.0
            running_mse_loss = 0.0

            for i, batch in tqdm(enumerate(data_loaders[phase]), total=data_lengths[phase] // batch_size,
                                 desc="Mini Batch {}".format(phase)):
                inputs = batch['image'].float().to(device)
                labels_mask = batch['mask'].float().to(device)
                labels_dist = batch['dist'][..., np.newaxis].float().to(device)
                optimizer.zero_grad()
                out_mask, out_dist = model(inputs)
                loss_mask = criterion_mask(out_mask, labels_mask)
                loss_dist = criterion_dist(out_dist, labels_dist)
                loss = (loss_coeff * loss_mask) + (1.0 - loss_coeff) * loss_dist

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_dice_loss += loss_mask.item()
                running_mse_loss += loss_dist.item()

                # print(running_loss)
            epoch_loss = running_loss / (data_lengths[phase] // batch_size)
            epoch_dice_loss = running_dice_loss / (data_lengths[phase] // batch_size)
            epoch_mse_loss = running_mse_loss / (data_lengths[phase] // batch_size)

            if phase == 'train':
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                train_loss = epoch_loss
            else:
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                val_loss = epoch_loss
            if writer:

                if phase == 'train':
                    tb_writer.add_scalar('Training_loss', epoch_loss, epoch)
                else:
                    tb_writer.add_scalar('Validation_loss', epoch_loss, epoch)

            history[phase].append(epoch_loss)
            history['{}_dice'.format(phase)].append(epoch_dice_loss)
            history['{}_mse'.format(phase)].append(epoch_mse_loss)
        history['epochs'].append(epoch)

        if epoch % model_checkpoint == (model_checkpoint - 1) or epoch == start_epoch + epochs - 1:
            model_filepath = os.path.join(task_folder_path,
                                          model_prefix + "_{}_dataset_{}epochs_{}coeff_mask.pkl".format(dataset_key,
                                                                                                        epoch + 1,
                                                                                                        loss_coeff))

            print("Save model checkpoint to: {}".format(model_filepath))

            torch.save(
                dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     train_loss=train_loss, val_loss=val_loss), model_filepath)

    history_filepath = os.path.join(task_folder_path,
                                    "history_" + model_prefix + "_{}epochs_{}coef.pkl".format(epochs, loss_coeff))
    print("Save history to {}".format(history_filepath))
    dump(history, open(history_filepath, 'wb'))

    print("Finished training")

    return history

def training_UNet(model, optimizer, criterion_mask, src_dir,
                         data_loaders, data_lengths, epochs, batch_size, model_checkpoint, task_folder_name, dev=0,
                         dataset_key="complete",
                         model_prefix="Trained_UNet_pytorch",
                         writer=None, notebook=None):
    task_folder_path = os.path.join(src_dir, "saved_models", task_folder_name)
    if not os.path.exists(task_folder_path):
        os.makedirs(task_folder_path)

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange

    if writer:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(src_dir, 'notebooks', 'runs',
                                               'UNet-{}_dataset_{}epochs.pkl'.format(dataset_key, epochs
                                                                                                   )))

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")

    model.to(device)
    history = create_history_multi_loss()

    for epoch in trange(epochs, desc="Training Epoch"):
        print(epoch + 1)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_dice_loss = 0.0

            for i, batch in tqdm(enumerate(data_loaders[phase]), total=data_lengths[phase] // batch_size,
                                 desc="Mini Batch {}".format(phase)):
                inputs = batch['image'].float().to(device)
                labels_mask = batch['mask'].float().to(device)
                optimizer.zero_grad()
                out_mask = model(inputs)
                loss_mask = criterion_mask(out_mask, labels_mask)
                loss = loss_mask

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_dice_loss += loss_mask.item()
                # print(running_loss)
            epoch_loss = running_loss / (data_lengths[phase] // batch_size)
            epoch_dice_loss = running_dice_loss / (data_lengths[phase] // batch_size)


            if phase == 'train':
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                train_loss = epoch_loss
            else:
                print('{} Loss: {:.4f})'.format(phase, epoch_loss))
                val_loss = epoch_loss
            if writer:

                if phase == 'train':
                    tb_writer.add_scalar('Training_loss', epoch_loss, epoch)
                else:
                    tb_writer.add_scalar('Validation_loss', epoch_loss, epoch)

            history[phase].append(epoch_loss)
            history['{}_dice'.format(phase)].append(epoch_dice_loss)

        history['epochs'].append(epoch)

        if epoch % model_checkpoint == (model_checkpoint - 1) or epoch == epochs - 1:
            model_filepath = os.path.join(task_folder_path,
                                          model_prefix + "_{}_dataset_{}epochs.pkl".format(dataset_key,
                                                                                                        epoch + 1))

            print("Save model checkpoint to: {}".format(model_filepath))

            torch.save(
                dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     train_loss=train_loss, val_loss=val_loss), model_filepath)

    history_filepath = os.path.join(task_folder_path,
                                    "history_" + model_prefix + "_{}epochs_.pkl".format(epochs))
    print("Save history to {}".format(history_filepath))
    dump(history, open(history_filepath, 'wb'))

    print("Finished training")

    return history