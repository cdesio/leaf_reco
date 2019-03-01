from math import ceil
import numpy as np
import pandas as pd
from collections import OrderedDict

try:  # old keras
    from keras.engine.topology import _to_list
except ImportError:  # new Keras
    from keras.utils.generic_utils import to_list as _to_list
from keras.utils import to_categorical
from numpy import concatenate as concat
from sklearn.utils.class_weight import compute_class_weight
#from export_train_test import (INDEX_TRAINING_KEY, INDEX_TEST_KEY, INDEX_VALIDATION_KEY)


def data_generator(fnames, batch_size=64, fdata=lambda X: X, ftarget=lambda y: to_categorical(y)):
    """
    Function to generate generator, according to the batch size(default=64)

    Parameters
    ----------
    fnames : list
          list of input filenames
    batch_size : np.int (default: 64)
          value of the batch size
    fdata
    ftarget
    Returns
    -------
    Examples
    --------
    Example usage of a classification problem generating categorical
    labels from cosz values (float), and TZ features from (batch, T,X,Y,Z) input tensor
    >>> def process_cosz(y):
    ...     y[y>0] = 1
    ...     y[y<=0] = 0
    ...     return to_categorical(y)
    >>> def get_TZ_only(X):
    ...     TZ = np.sum(X, axis=(2, 3))
    ...     TZ = TZ[:, np.newaxis, ...]
    ...     return TZ
    >>> data_generator(fnames, batch_size=64, fdata=get_TZ_only, ftarget=process_cosz)
    Example usage for regression problem with no change to labels
    >>> data_generator(fnames, batch_size=64, ftarget=lambda y: y)
    Example usage for generating multiple batches of data for multi-input networks
    >>> from keras import backend as K
    >>> def get_TZXY_data(X):
    ...     TZ = np.sum(X, axis=(2, 3))
    ...     XY = np.sum(X, axis=(1, 4))
    ...     if K.image_data_format() == "channels_first":
    ...         TZ = TZ[:, np.newaxis, ...]
    ...         XY = XY[:, np.newaxis, ...]
    ...     else:
    ...         TZ = TZ[..., np.newaxis]
    ...         XY = XY[..., np.newaxis]
    ...     return [TZ, XY]
    >>> data_generator(fnames, batch_size=64, fdata=get_TZXY_data)
    """

    while True:
        file_idx = 0
        X_buff = Y_buff = None
        residual = False
        while file_idx < len(fnames):
            fname = fnames[file_idx]
            Xy = np.load(fname)
            X, y = Xy['x'], Xy['y']
            Y = ftarget(y)
            X = _to_list(fdata(X))  # X will be finally a list
            idx = 0  # batch current file
            while idx < Y.shape[0]:
                if residual:  # i.e. there are samples stored from previous iteration
                    incr = (batch_size - X_buff[0].shape[0])
                else:
                    incr = batch_size

                start, end = idx, idx + incr

                if end > Y.shape[0]:  # current file is completed
                    if X_buff is None:
                        X_buff, Y_buff = [X_in[start:] for X_in in X], Y[start:]
                    else:
                        X_buff = [concat((X_in_buff, X_in[start:])) for X_in_buff, X_in in zip(X_buff, X)]
                        Y_buff = concat((Y_buff, Y[start:]))
                    residual = True
                    break
                X_batch, Y_batch = [X_in[start:end] for X_in in X], Y[start:end]
                if residual:
                    X_batch = [concat((X_in_buff, X_in_batch)) for X_in_buff, X_in_batch in zip(X_buff, X_batch)]
                    Y_batch = concat((Y_buff, Y_batch))
                else:
                    X_buff = Y_buff = None

                yield X_batch, Y_batch
                idx += incr
                residual = False
            file_idx += 1
        else:
            if residual:
                yield X_buff, Y_buff


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def get_n_iterations(fnames_list, target_key="y", batch_size=64):
    """
    Function to get the number of iterations required to
    process the total set of samples extracted from the list
    of data files in input.
    Parameters
    ----------
    fnames_list: list
        List of data files to read from. Expected file format is Numpy/Compressed
        as created from `numpy.savez_compressed`.
    target_key: str (default "y")
        The key in the file to be used to calculate the number of samples
    batch_size: int (default 64)
        The size of the batch that is expected to be used in the training process.
    Returns
    -------
        int, int: total number of iterations and total number of events
        extracted from list of input files.
    """
    tot_events = 0
    print(fnames_list)
    for fil in fnames_list:
        yf = np.load(fil)[target_key]
        print(fil)
        print(yf.shape)
        tot_events += yf.shape[0]
    iterations = int(ceil(tot_events / float(batch_size)))
    return iterations, tot_events


def get_class_weights(fnames_list, target_key="y"):
    """
    Function to get class weights for samples extracted from the list
    of data files in input.
    Parameters
    ----------
    fnames_list: list
        List of data files to read from. Expected file format is Numpy/Compressed
        as created from `numpy.savez_compressed`.
    target_key: str (default "y")
        The key in the file to be used to calculate the number of samples
    Returns
    -------
    class_weights: dict
        A dictionary mapping each class to its corresponding weight.
    """
    y_all = None
    for fil in fnames_list:
        yf = np.load(fil)[target_key]
        if y_all is None:
            y_all = yf
        else:
            y_all = np.concatenate((y_all, yf))
    class_weight_vect = compute_class_weight('balanced', np.unique(y_all), y_all)
    return {i: w for i, w in enumerate(class_weight_vect)}
