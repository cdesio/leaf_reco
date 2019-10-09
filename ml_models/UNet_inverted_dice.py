import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K
IMG_WIDTH = 1400
IMG_HEIGHT = 1400

IMG_WIDTH = 1400
IMG_HEIGHT = 1400

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def get_unet(img_rows = IMG_HEIGHT, img_cols = IMG_WIDTH):
    """"""
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), name='convT_5')(conv5), conv4], 
                      axis=-1, name='up_convT5_conv4')
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_1')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_2')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='convT_6')(conv6), conv3], 
                      axis=-1, name='up_convT6_conv3')
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7_1')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7_2')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='convT_7')(conv7), conv2], 
                      axis=-1, name='up_convT7_conv2')
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv8_1')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv8_2')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='convT_8')(conv8), conv1], 
                      axis=-1, name='up_convT8_conv1')
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv9_1')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv9_2')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv10_sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
#adam ls=1e-5, RMSprop(lr=2e-4)
    return model
