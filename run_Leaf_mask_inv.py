#!/usr/bin/python
import os
import numpy as np
import re
import sys

in_folder = sys.argv[1]
dist = sys.argv[2]
out_folder = sys.argv[3]
regex = re.compile(r'\d+')

fnb = [np.int(regex.findall(i)[0]) for i in os.listdir(in_folder) if "mask" not in i and str(i).startswith("File")]
print(in_folder, dist, out_folder)
for nb in sorted(fnb):
    #print(nb)
    os.system("python /Users/cdesio/UoB/MAPS/leaf_reco/Leaf_mask_inv.py {} {} {} {}".format(in_folder, nb, out_folder, dist))
