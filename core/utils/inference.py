import numpy as np
import torch


def inference_phase_rUNet(model, data_loaders, data_lengths, batch_size, dev=0, notebook=None, test=True):
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")

    # model = cUNet(out_size=1)

    # model.load_state_dict(torch.load(model_name))
    model.eval()
    model.to(device);

    y_true = []
    y_pred = []
    if test:
        for i, batch in tqdm(enumerate(data_loaders["test"]), total=data_lengths["test"] // batch_size, desc="Batch"):
            true_images, true_dists = batch["image"], batch["dist"]
            _, pred_dists = model(true_images.float().to(device))
            for j, (img, tr_dist, pr_dist) in enumerate(zip(true_images,
                                                            true_dists.cpu().detach().numpy(),
                                                            pred_dists.cpu().detach().numpy())):
                y_true.append(tr_dist)
                y_pred.append(pr_dist)
    else:
        for i, batch in tqdm(enumerate(data_loaders), total=data_lengths // batch_size, desc="Batch"):
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


def inference_phase_rUNet_plot_notebook(model, data_loaders, data_lengths, batch_size, stop=1, dev=0, test=True):
    from tqdm.notebook import tqdm
    import matplotlib.pyplot as plt

    device = torch.device("cuda:{}".format(dev) if torch.cuda.is_available() else "cpu")
    # model = cUNet(out_size=1)

    # model.load_state_dict(torch.load(model_name))
    model.eval()
    model.to(device);
    if test:
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
        else:
            for i, batch in tqdm(enumerate(data_loaders), total=data_lengths // batch_size,
                                 desc="Batch"):

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