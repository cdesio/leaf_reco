#!/usr/bin/python 
import os
import numpy as np
import re

in_folders = []
out_folders = []

for i in os.listdir("/Users/yw18581/temp"):
    if str(i).startswith("10x10"):
        if str(i).endswith("clean"):
            out_folders.append(i) 
        else:
            in_folders.append(i)

regex = re.compile(r'\d+')


for i, j in zip(sorted(in_folders), sorted(out_folders)):
    infolder, outfolder, dist = (i, j, regex.findall(i)[2])
    os.system("python /Users/yw18581/physics/leaf_reco/run_Leaf_mask_inv_selection.py /Users/yw18581/temp/{} {} /Users/yw18581/temp/{}".format(infolder,dist, outfolder))
    print("masks produced and files moved.")
    os.system("rm -rf {}.".format(infolder))
    os.system("mv {} /Volumes/TMBackups/second_batch/. ".format(outfolder))

print("done")
os.system("tar -cvf /Volumes/TMBackups/second_batch.tar /Volumes/TMBackups/second_batch")


