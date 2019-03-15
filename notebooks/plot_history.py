import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, ymin_loss=None, ymax_loss=None, ymin_acc=None, ymax_acc=None):
    if ymin_acc and ymin_acc is None:
        ymin_loss=np.min(history.history['loss'])
        ymax_loss=np.max(history.history['loss'])
    f1 = plt.figure()
    plt.plot(history.history['loss'],label='training')
    plt.plot(history.history['val_loss'],label='validation')
    plt.ylim(ymin_loss,ymax_loss)
    plt.legend(loc='upper right')
    plt.show()
    if 'acc' in history.history:
        if ymin_acc and ymin_acc is None:
            ymin_acc=np.min(history.history['acc'])
            ymax_acc=np.max(history.history['acc'])
        f2 = plt.figure()
        plt.plot(history.history['acc'],label='training')
        plt.plot(history.history['val_acc'],label='validation')
        plt.ylim(ymin_acc,ymax_acc)
        plt.legend(loc='upper right')
        plt.show()
    return