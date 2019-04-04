import numpy as np
import matplotlib; matplotlib.use('agg')
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)
import sys 
import os
f_no = sys.argv[1]

data_dir = '/storage/yw18581/data'
img_path =os.path.join(data_dir, '10x10_30mm_v1', 'File_{}.tiff'.format(f_no))
image = imread(img_path)
print(img_path)
print("import image and select profile")

def calculate_profile(image):
    # Initial level set
    init_ls =  checkerboard_level_set(image.shape, 5)
    # List with intermediate results for plotting the evolution

    ls = morphological_chan_vese(image, 10, init_level_set=init_ls, smoothing=5)
    return ls

ls= calculate_profile(image)
print("select leaf")
def select_profile(array):
    sel=[]
    for i in range(np.max(np.argwhere(array)[:,0])):
        sel.append([i,np.max(np.argwhere(array)[:,1][np.argwhere(array)[:,0]==i])])
        #print(i,np.max(np.argwhere(array)[:,1][np.argwhere(array)[:,0]==i]))
    sel = np.asarray(sel)
    return sel

leaf = select_profile(ls[1000:1280])
leaf_position = np.min(leaf[:,1])
print("calculated leaf position:{}".format(leaf_position))

if leaf_position>=2300:
    ls_inv=(~ls.astype(bool)).astype(int)
    leaf = select_profile(ls_inv[1000:1280])
    leaf_position = np.min(leaf[:,1])
    print("new leaf position: {}".format(leaf_position)) 
    
plt.figure(figsize=(2400/96, 2800/96), dpi=96)
plt.style.use('dark_background')
plt.axes([0,0,1,1], frameon=False)
plt.plot(leaf[:,1], leaf[:,0]+1000, c='w', alpha=1)
plt.fill_betweenx(leaf[:,0]+1000, leaf[:,1],x2=2205,color='w',alpha=1)
fig = plt.imshow(image, cmap="gray", alpha=0)
plt.ylim(2800,0)
plt.xlim(0,2400)
plt.box(False)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#plt.savefig("mask_only.tiff")
plt.savefig(os.path.join(data_dir, "10x10_30mm_v1",'File_{}_30mm_mask_{}.tiff'.format(f_no,leaf_position)))
plt.close('all')