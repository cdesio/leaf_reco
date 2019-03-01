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

from keras import backend as K

TX_AXIS = (3, 4)
TY_AXIS = (2, 4)
TZ_AXIS = (2, 3)


def direction_net_data_generator(fnames, batch_size=64):
    """
    """

    def get_Time_Coord(X, sum_axis):
        TC = np.sum(X, axis=sum_axis)
        if K.image_data_format() == "channels_first":
            TC = TC[:, np.newaxis, ...]
        else:
            TC = TC[..., np.newaxis]
        return TC.astype(K.floatx())

    while True:
        file_idx = 0
        X_buff = Dirx_buff = Diry_buff = Dirz_buff = None
        residual = False
        while file_idx < len(fnames):
            fname = fnames[file_idx]
            Xy = np.load(fname)
            X, Dirx, Diry, Dirz = Xy['x'], Xy['dirx'], Xy['diry'], Xy['dirz']
            Tx = get_Time_Coord(X, sum_axis=TX_AXIS)
            Ty = get_Time_Coord(X, sum_axis=TY_AXIS)
            Tz = get_Time_Coord(X, sum_axis=TZ_AXIS)
            X = [Tx, Ty, Tz]
            n_samples = X[0].shape[0]
            idx = 0  # batch current file
            while idx < n_samples:
                if residual:  # i.e. there are samples stored from previous iteration
                    incr = (batch_size - X_buff[0].shape[0])
                else:
                    incr = batch_size

                start, end = idx, idx + incr

                if end > n_samples:  # current file is completed
                    if X_buff is None:
                        X_buff = [X_in[start:] for X_in in X]
                        Dirx_buff, Diry_buff, Dirz_buff = Dirx[start:], Diry[start:], Dirz[start:]
                    else:
                        X_buff = [concat((X_in_buff, X_in[start:])) for X_in_buff, X_in in zip(X_buff, X)]
                        Dirx_buff = concat((Dirx_buff, Dirx[start:]))
                        Diry_buff = concat((Diry_buff, Diry[start:]))
                        Dirz_buff = concat((Dirz_buff, Dirz[start:]))
                    residual = True
                    break
                X_batch = [X_in[start:end] for X_in in X]
                Dirx_batch = Dirx[start:end]
                Diry_batch = Diry[start:end]
                Dirz_batch = Dirz[start:end]
                if residual:
                    X_batch = [concat((X_in_buff, X_in_batch)) for X_in_buff, X_in_batch in zip(X_buff, X_batch)]
                    Dirx_batch = concat((Dirx_buff, Dirx_batch))
                    Diry_batch = concat((Diry_buff, Diry_batch))
                    Dirz_batch = concat((Dirz_buff, Dirz_batch))
                else:
                    X_buff = Dirx_buff = Diry_buff = Dirz_buff = None

                Dir_Euclidean = np.hstack((Dirx_batch[..., np.newaxis],
                                           Diry_batch[..., np.newaxis],
                                           Dirz_batch[..., np.newaxis])).astype(K.floatx())
                ones_batch = (Dirx_batch ** 2) + (Diry_batch ** 2) + (Dirz_batch ** 2)
                ones_batch = ones_batch.astype(K.floatx())
                Dirx_batch = Dirx_batch.astype(K.floatx())
                Diry_batch = Diry_batch.astype(K.floatx())
                Dirz_batch = Dirz_batch.astype(K.floatx())
                yield X_batch, [Dirx_batch, Diry_batch, Dirz_batch, Dir_Euclidean, ones_batch]
                idx += incr
                residual = False
            file_idx += 1
        else:
            if residual:
                Dir_Euclidean = np.hstack((Dirx_buff[..., np.newaxis],
                                           Diry_buff[..., np.newaxis],
                                           Dirz_buff[..., np.newaxis])).astype(K.floatx())
                ones_batch = (Dirx_buff ** 2) + (Diry_buff ** 2) + (Dirz_buff ** 2)
                ones_batch = ones_batch.astype(K.floatx())
                Dirx_buff = Dirx_buff.astype(K.floatx())
                Diry_buff = Diry_buff.astype(K.floatx())
                Dirz_buff = Dirz_buff.astype(K.floatx())
                yield X_buff, [Dirx_buff, Diry_buff, Dirz_buff, Dir_Euclidean, ones_batch]


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def metadata_generator(index_filelist, xy_filelist, metadata_keylist,
                       index_key=INDEX_TEST_KEY, batch_size=64):
    """
    Function to generate a set of current_file_metadata associated to each batch of events (identified by the
    corresponding index set, namely "training", "validation", "test").
    Parameters
    ----------
    index_filelist: list or tuple
        List of paths to files containing indices for events
    xy_filelist: list or tuple
        List of paths to files containing current_file_metadata so that
        len(xy_filelist) == len(index_filelist)
        If current_file_metadata are organised in multiple files, a list of tuples
        must be provided, and current_file_metadata will be stacked accordingly.
    metadata_keylist: list or tuple
        List of keys to extract for each event from the current_file_metadata file
    index_key: str (default: "test")
        The key identifying the set of indices to consider
    batch_size: int (default:64)
        The size of the batch
    Returns
    -------
    current_file_metadata: pd.DataFrame
        DataFrame containing current_file_metadata ...
    """

    if index_key not in (INDEX_TEST_KEY, INDEX_VALIDATION_KEY, INDEX_TRAINING_KEY):
        raise ValueError('The index key should be one of the following: %s, %s, %s' % (INDEX_TEST_KEY,
                                                                                       INDEX_VALIDATION_KEY,
                                                                                       INDEX_TRAINING_KEY))

    def _create_dataframe():
        md_batch = OrderedDict()
        for key in metadata_keylist:
            md_batch[key] = current_file_metadata[key][idx_batch]
        md_batch['file_evt_index'] = file_idx
        md_batch['evt_index'] = idx_batch
        return pd.DataFrame(md_batch)

    while True:
        file_idx = 0
        df_metadata_buff, df_metadata_batch = None, None
        residual = False
        while file_idx < len(index_filelist):
            index_fname = index_filelist[file_idx]
            # Get set of indices
            indices = np.load(index_fname)[index_key]

            metadata_fnames = xy_filelist[file_idx]
            if not isinstance(metadata_fnames, tuple):
                metadata_fnames = (metadata_fnames,)

            # load current_file_metadata
            current_file_metadata = {}
            for key in metadata_keylist:
                md_key = None
                for path in metadata_fnames:
                    if md_key is None:
                        md_key = np.load(path)[key]
                    else:
                        md_key = np.hstack((md_key, np.load(path)[key]))

                assert np.all(np.isin(indices, np.arange(md_key.shape[0])) == True)
                current_file_metadata[key] = md_key

            idx = 0  # batch current file
            idx_batch = None
            while idx < indices.shape[0]:
                if residual:  # i.e. there are samples stored from previous iteration
                    incr = (batch_size - df_metadata_buff.shape[0])
                else:
                    incr = batch_size
                start, end = idx, idx + incr

                if end > indices.shape[0]:  # current file is completed
                    idx_batch = indices[start:]
                    df = _create_dataframe()
                    if df_metadata_buff is None:
                        df_metadata_buff = df
                    else:
                        df_metadata_buff = pd.concat((df_metadata_buff, df))
                    residual = True
                    break

                idx_batch = indices[start:end]
                df_metadata_batch = _create_dataframe()
                if residual:
                    df_metadata_batch = pd.concat((df_metadata_buff, df_metadata_batch))
                else:
                    df_metadata_buff = None

                yield df_metadata_batch
                idx += incr
                residual = False
            file_idx += 1
        else:
            if residual:
                yield df_metadata_buff


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
    for fil in fnames_list:
        yf = np.load(fil)[target_key]
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