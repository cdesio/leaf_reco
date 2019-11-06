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


def select_profile(array):
    sel = []
    for i in range(np.max(np.argwhere(array)[:, 0])):
        sel.append([i, np.max(np.argwhere(array)[:, 1][np.argwhere(array)[:, 0] == i])])
        # print(i,np.max(np.argwhere(array)[:,1][np.argwhere(array)[:,0]==i]))
    sel = np.asarray(sel)
    return sel


def select_profile_2(array):
    arr = np.argwhere(array)
    y = arr[:, 0]
    X = arr[:, 1]
    y_uniques = np.unique(y)
    y_out = [np.max(X[y == yi]) for yi in y_uniques]
    return np.column_stack((y_uniques, y_out))


# leaf = select_profile(ls[1000:1280])
leaf = select_profile_2(ls[1000:1280])
leaf_position = np.min(leaf[:, 1])
print("calculated leaf position:{}".format(leaf_position))

if leaf_position >= 2300:
    ls_inv = (~ls.astype(bool)).astype(int)
    leaf = select_profile_2(ls_inv[1000:1280])
    leaf_position = np.min(leaf[:, 1])
    print("new leaf position: {}".format(leaf_position))
# test1
# for i, (j, k) in enumerate(leaf):
#    if k>leaf[i-1][1]+100:
#        leaf[i][1]=leaf[i-1][1] 
# test2
# for i, (j, k) in enumerate(leaf):
#    if i >3 and i < len(leaf)-3: 
#        if np.abs(k-leaf[i-1][1])>150:
#            leaf[i][1]=np.mean([leaf[i-3][1], leaf[i+3][1]])
# test3
index = []
for i, (j, k) in enumerate(leaf):
    if i < len(leaf) - 1:
        if np.abs(k - leaf[i + 1][1]) > 200 or np.abs(k - leaf[i + 1][1]) == 0:
            if leaf[i][1] >= leaf[i + 1][1]:
                index.append(i)
            elif leaf[i + 1][1] > leaf[i][1]:
                index.append(i + 1)
leaf = np.delete(leaf, index, axis=0)

print("check the borders")
if np.abs(leaf[-1][1] - leaf[-2][1] >= 200):
    print("found it")
    if leaf[-1][1] > leaf[-2][1]:
        print("and change it")
        leaf[-1][1] = leaf[-2][1]
        print(leaf[-1][1])
print("done")

# end test 3
plt.figure(figsize=(2400 / 96, 2800 / 96), dpi=96)
plt.style.use('dark_background')
plt.axes([0, 0, 1, 1], frameon=False)
plt.plot(leaf[:, 1], leaf[:, 0] + 1000, c='w', alpha=1)
plt.fill_betweenx(leaf[:, 0] + 1000, leaf[:, 1], x2=2205, color='w', alpha=1)
fig = plt.imshow(image, cmap="gray", alpha=0)
plt.ylim(2800, 0)
plt.xlim(0, 2400)
plt.box(False)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
# plt.savefig("mask_only.tiff")
plt.savefig(os.path.join(out_folder, '{}_{}mm_mask_{}.tiff'.format('_'.join(split), dist, leaf_position)))
plt.close('all')
