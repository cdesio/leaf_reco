
import os
import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import argparse

regex = re.compile(r'\d+')

IMG_WIDTH = 1400
IMG_HEIGHT = 1400
ROW_SLICE = slice(0, 1400)
COL_SLICE = slice(1000, None)

DATA_DIR_IH = "/data/uob"
DATA_DIR_DEEPTHOUGHT = "/storage/yw18581/data"
data_dir = DATA_DIR_DEEPTHOUGHT

source_data_folder = data_dir
print(data_dir)
print(source_data_folder)
TRAIN_VAL_TEST_DIR = os.path.join(data_dir, "train_validation_test")

if not os.path.exists(TRAIN_VAL_TEST_DIR):
    os.makedirs(TRAIN_VAL_TEST_DIR)

def get_filename_and_data_image(folder):
    dist_selection = np.ravel([regex.findall(i) for i in folder.split('_') if i.endswith('mm')])[0]
    images_list = [[os.path.join(folder, "{}".format(filename)), dist_selection]
                   for filename in sorted(os.listdir(folder)) if "mask" not in filename and filename.startswith("File")]
    df_images = pd.DataFrame.from_records(images_list, columns=('path', 'dist'))
    return df_images

def get_filename_and_data_mask(folder):
    dist_selection = np.ravel([regex.findall(i) for i in folder.split('_') if i.endswith('mm')])[0]
    masks_list = [[os.path.join(folder, "{}".format(filename)),
                   dist_selection]
                  for filename in sorted(os.listdir(folder)) if "mask" in filename and filename.startswith("File")]
    df_masks = pd.DataFrame.from_records(masks_list, columns=('path', 'dist'))
    return df_masks

def train_validation_test(df, stratification_key=None):
    if stratification_key is not None:
        stratify_arr = df[stratification_key].values
    else:
        stratify_arr = None

    indices = np.arange(len(df))
    training_indices, test_indices = train_test_split(indices, test_size=0.20,
                                                   random_state=42, stratify=stratify_arr)

    if stratification_key is not None:
        stratify_arr = df.iloc[training_indices][stratification_key]

    train_v_indices, val_indices = train_test_split(training_indices, test_size=0.20,
                                                    random_state=42, stratify=stratify_arr)
    return training_indices, test_indices, train_v_indices, val_indices


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to export train, validation and test data')

    parser.add_argument('-i', '--i', help='input string: names of input data folders', required=True)
    parser.add_argument('-o', '--o', help='output string', required=True)

    args = parser.parse_args()
    print(args.i, args.o)

    ## Loading data
    print("Loading data filenames")
    folders = args.i.split(" ")

    #folder_2mm = os.path.join(source_data_folder, "10x10_2mm_8bit")
    #folder_4mm = os.path.join(source_data_folder, "10x10_4mm_v2 copy")
    #folder_10mm = os.path.join(source_data_folder, "10x10_10mm_v2_8bit")
    #folder_20mm = os.path.join(source_data_folder, "10x10_20mm_v1")
    #folder_30mm = os.path.join(source_data_folder, "10x10_30mm_v1")

    #fnames_orig, fnames_mas = get_filename_and_data(folder_path)
    #fnames_orig_2mm, fnames_mask_2mm = get_filename_and_data(folder_2mm)
    #fnames_orig_4mm, fnames_mask_4mm = get_filename_and_data(folder_4mm)
    #fnames_orig_10mm, fnames_mask_10mm = get_filename_and_data(folder_10mm)
    #fnames_orig_20mm, fnames_mask_20mm = get_filename_and_data(folder_20mm)
    #fnames_orig_30mm, fnames_mask_30mm = get_filename_and_data(folder_30mm)

    #print("check number of files per type")
    #print(len(fnames_mask_2mm), len(fnames_mask_4mm), len(fnames_mask_10mm),
    #      len(fnames_mask_20mm), len(fnames_mask_30mm))
    #print(len(fnames_orig_2mm), len(fnames_orig_4mm), len(fnames_orig_10mm),
    #      len(fnames_orig_20mm), len(fnames_orig_30mm))

    df_orig = pd.concat([get_filename_and_data_image(folder) for folder in folders])
    df_mask = pd.concat([get_filename_and_data_mask(folder) for folder in folders])

    #df_orig = pd.concat([fnames_orig_2mm, fnames_orig_4mm, fnames_orig_10mm,
    #                     fnames_orig_20mm, fnames_orig_30mm], ignore_index=True)
    #df_mask = pd.concat([fnames_mask_2mm, fnames_mask_4mm, fnames_mask_10mm,
    #                     fnames_mask_20mm, fnames_mask_30mm], ignore_index=True)
    print("check total number of files")
    print("original images:{}, mask files:{}".format(len(df_orig), len(df_mask)))

    print("Train test split")




    _, test_indices, train_indices, val_indices = train_validation_test(df_mask, 
                                                                stratification_key='dist')

    print("Train dataset:{} files, Validation dataset:{}, Test dataset:{} files".format(len(train_indices), 
                                                                                        len(val_indices),
                                                                                        len(test_indices)))



    df_mask_train = df_mask.iloc[train_indices]
    df_mask_test = df_mask.iloc[test_indices]
    df_mask_val = df_mask.iloc[val_indices]

    print("Import data (this may take some time)")

    X_train = np.asarray([imread(df_orig['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in train_indices])
    X_test = np.asarray([imread(df_orig['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in test_indices])
    X_val = np.asarray([imread(df_orig['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in val_indices])

    y_train = np.asarray([imread(df_mask['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in train_indices])
    y_test = np.asarray([imread(df_mask['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in test_indices])
    y_val = np.asarray([imread(df_mask['path'].iloc[i])[ROW_SLICE, COL_SLICE] for i in val_indices])

    metadata_train = np.asarray(df_orig['dist'].iloc[train_indices], dtype=int)
    metadata_test = np.asarray(df_orig['dist'].iloc[test_indices], dtype=int)
    metadata_val = np.asarray(df_orig['dist'].iloc[val_indices], dtype=int)

    print("Train, validation, test data shape")

    print(X_train.shape, y_train.shape, metadata_train.shape,
          X_val.shape, y_val.shape, metadata_val.shape,
          X_test.shape, y_test.shape, metadata_test.shape)

    print("Add new dimension - for channels last data format in keras")
    X_train = X_train[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    y_val = y_val[..., np.newaxis]

    out_str = args.o
    train_out_str = 'Xy_train_'+out_str+'.npz'
    val_out_str = 'Xy_val_'+out_str+'.npz'
    test_out_str = 'Xy_test_'+out_str+'.npz'
    print("Save train, validation and test data to output files in {} , {} and {}".format(os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                                       train_out_str),
                                                                                          os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                                       val_out_str),
                                                                                          os.path.join(TRAIN_VAL_TEST_DIR,
                                                                                                       test_out_str)))

    np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR, train_out_str),
                        x=X_train, y=y_train, dist=metadata_train)
    np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR, test_out_str),
                        x=X_test, y=y_test, dist=metadata_test)
    np.savez_compressed(os.path.join(TRAIN_VAL_TEST_DIR, val_out_str),
                        x=X_val, y=y_val, dist=metadata_val)
