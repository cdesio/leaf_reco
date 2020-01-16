import numpy as np
import matplotlib;

matplotlib.use('agg')
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)
import sys
import os
import re

regex = re.compile(r'\d+')
in_folder = sys.argv[1]
f_no = sys.argv[2]
dist = sys.argv[3]
out_folder = sys.argv[4]
# data_dir = '/Users/cdesio/UoB/Jordan'
fname_key = sys.argv[5]
split = fname_key.split("_")
split.append(str(f_no))
img_path = os.path.join(in_folder, '{}.tiff'.format('_'.join(split)))
image = imread(img_path)
print(img_path)
print("import image and select profile")

def calculate_profile(image):
    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 5)
    # List with intermediate results for plotting the evolution

    ls = morphological_chan_vese(image, 10, init_level_set=init_ls, smoothing=5)
    return ls

ls = calculate_profile(image)
print("select leaf")

def select_profile(array, lower=450, upper=2200):
    arr = np.argwhere(array[lower:upper])
    y = arr[:, 0]
    X = arr[:, 1]
    y_uniques = np.unique(y)
    y_out = [np.max(X[y == yi]) for yi in y_uniques]
    leaf = np.column_stack((y_uniques+lower, y_out))
    plt.plot(leaf[:,1], leaf[:,0], c='r')
    plt.xlim(0,2400)
    plt.ylim(2800,0)
    plt.show()
    return leaf

range_list =[(430, 700),
             (1000, 1300),
             (1450, 1800),
             (1900, 2240)]

def select_multiple_profiles(ls, range_list):
    leaves = []
    for low, up in range_list:
        leaf = select_profile(ls,low, up)
        leaf_position = np.min(leaf[:,1])
        if leaf_position>=2300:
            ls_inv = (~ls.astype(bool)).astype(int)
            leaf_inv = select_profile(ls_inv)
            leaves.append(leaf_inv)
        else:
            leaves.append(leaf)
    return leaves

def check_borders(leaves):
    for leaf in leaves:
        index = []
        for i, (j, k) in enumerate(leaf):
            if i < len(leaf) - 1:
                if np.abs(k - leaf[i + 1][1]) > 200 or np.abs(k - leaf[i + 1][1]) == 0:
                    if leaf[i][1] >= leaf[i + 1][1]:
                        index.append(i)
                    elif leaf[i + 1][1] > leaf[i][1]:
                        index.append(i + 1)
        leaf = np.delete(leaf, index, axis=0)

        if np.abs(leaf[-1][1] - leaf[-2][1] >= 200):
            print("found it")
            if leaf[-1][1] > leaf[-2][1]:
                print("and change it")
                leaf[-1][1] = leaf[-2][1]
                print(leaf[-1][1])
        print("done")
    return leaves


plt.figure(figsize=(2400 / 96, 2800 / 96), dpi=96)
plt.style.use('dark_background')
plt.axes([0, 0, 1, 1], frameon=False)
for leaf in leaves:
    plt.plot(leaf[:, 1], leaf[:, 0], c='w', alpha=1)
    plt.fill_betweenx(leaf[:, 0], leaf[:, 1], x2=2340, color='w', alpha=1)
fig = plt.imshow(image, cmap="gray", alpha=0)
plt.ylim(2800, 0)
plt.xlim(0, 2400)
plt.box(False)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#plt.show()
# plt.savefig("mask_only.tiff")
plt.savefig(os.path.join(out_folder, '{}_{}mm_mask_{}.tiff'.format('_'.join(split), dist, leaf_position)))
plt.close('all')