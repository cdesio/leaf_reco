#!/usr/bin/python
import os
import numpy as np
import re
import sys
    
    def run_leaf_mask_inv(in_folder, dist, out_folder):
    
    regex = re.compile(r'\d+')

    f_clean = [np.int(regex.findall(i)[0]) for i in os.listdir(out_folder) if "mask" in i and str(i).startswith("File")]
    f_tot = [np.int(regex.findall(i)[0]) for i in os.listdir(in_folder) if "mask" in i and str(i).startswith("File")]
    fbad = [nb for nb in f_tot if nb not in f_clean]
    print(f_clean, f_tot, fbad)
    print("files in clean dir:{}, files in original dir:{}, files to process:{}".format(len(f_clean), len(f_tot), len(fbad)))

    print(in_folder, dist, out_folder)
    for nb in sorted(fbad):
     #print(nb)
        os.system("python /storage/yw18581/src/leaf_reco/Leaf_mask_inv.py {} {} {} {}".format(in_folder, nb, dist, out_folder))
    print("masks created!")
    print("copy remaining input files to clean dir")
    for i in sorted(os.listdir(in_folder)):
        if "mask" not in i:
            os.system("cp {}/{} {}/".format(in_folder, i, out_folder))
    return

if __name__ == '__main__':
    in_folder = sys.argv[1]
    dist = sys.argv[2]
    out_folder = sys.argv[3]

    print("done")
