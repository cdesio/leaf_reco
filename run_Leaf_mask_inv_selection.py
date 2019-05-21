#!/usr/bin/python
import os
import numpy as np
import re
import sys

in_folder = sys.argv[1]
dist = sys.argv[2]
out_folder = sys.argv[3]
regex = re.compile(r'\d+')

f_clean = [np.int(regex.findall(i)[0]) for i in os.listdir(out_folder) if "mask" in i and str(i).startswith("File")]
f_tot = [np.int(regex.findall(i)[0]) for i in os.listdir(in_folder) if "mask" in i and str(i).startswith("File")]
fbad = [nb for nb in f_tot if nb not in f_clean]
print(in_folder, dist, out_folder)
for nb in sorted(fbad):
    #print(nb)
    os.system("python /Users/cdesio/UoB/MAPS/leaf_reco/Leaf_mask_inv.py {} {} {} {}".format(in_folder, nb, dist, out_folder))
