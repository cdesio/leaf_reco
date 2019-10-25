import os
import sys
#import re

if __name__ == '__main__':

    if not len(sys.argv)==3:
        raise Exception("Please insert folder name and out_folder")

    #regex = re.compile(r'\d+')

    DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"
    data_dir = DATA_DIR_DEEPTHOUGHT
    dataset = os.path.join(data_dir, "dataset")

    infolder_path = sys.argv[1]
    outfolder_path = sys.argv[2]

    infiles = [file for file in os.listdir(infolder_path) if "mask" not in file and str(file).startswith("File")]
    for f in infiles:
        print("copying {} to {}".format(f, outfolder_path))
        os.system("cp {} {}".format(os.path.join(infolder_path,f), outfolder_path))
    print("done")