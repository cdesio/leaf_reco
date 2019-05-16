#!/usr/bin/python
import os
import numpy as np
import re
import sys

folder = sys.argv[1]

regex = re.compile(r'\d+')

fnb = [np.int(regex.findall(i)[0]) for i in os.listdir(folder) if "mask" not in i and str(i).startswith("File")]

for nb in fnb:
    os.system("python /Users/cdesio/UoB/MAPS/leaf_reco/Leaf_mask_inv.py {}".format(nb))
