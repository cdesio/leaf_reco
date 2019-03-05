"""
Deep Neural Networks implementations in Keras & TensorFlow backend
"""

from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import AveragePooling3D, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Lambda, Permute, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.engine import Layer

from keras.models import Model
from keras.losses import categorical_crossentropy, mse
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
try: #old Keras
    from keras.applications.resnet50 import conv_block, identity_block
except ImportError: # new Keras
    from keras_applications.resnet50 import conv_block, identity_block

from keras import backend as K
from os import path as p
from os import makedirs
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

CHECKPOINT_FOLDER_PATH = p.join(p.abspath(p.curdir), 'model')

if K.image_data_format() == 'channels_first':
    TZ_SHAPE = (1, 75, 18)
    TX_SHAPE = (1, 75, 16)
    TY_SHAPE = (1, 75, 15)
    XY_SHAPE = (1, 16, 15)
    TXYZ_SHAPE_TIME_DISTRIBUTED = (75, 1, 16, 15, 18)
    LEAF_SHAPE = (1, 1400, 1400)
elif K.image_data_format() == 'channels_last':
    TZ_SHAPE = (75, 18, 1)
    TX_SHAPE = (75, 16, 1)
    TY_SHAPE = (75, 15, 1)
    XY_SHAPE = (16, 15, 1)
    TXYZ_SHAPE_TIME_DISTRIBUTED = (75, 16, 15, 18, 1)
    LEAF_SHAPE = (1400, 1400, 1)
else:
    raise ValueError('Please check Keras configuration file for Image data format')

DEFAULT_OPT = Adadelta()  # Default optimizer for networks


class PTanh(Layer):
    """Parametric version of an Hyperbolic Tangent (tanh).
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        p: float >= 0. slope coefficient.
    """

    def __init__(self, alpha=8.0, **kwargs):
        super(PTanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs, **kwargs):
        tanh_inputs = inputs / self.alpha
        return self.alpha * K.tanh(tanh_inputs)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(PTanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PSigmoid(Layer):
    """Parametric version of an Hyperbolic Tangent (tanh).
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        p: float >= 0. slope coefficient.
    """

    def __init__(self, alpha=8.0, beta=4.0, **kwargs):
        super(PSigmoid, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs, **kwargs):
        pos = K.relu(inputs)
        pos = self.alpha * (K.sigmoid(pos) - K.cast_to_floatx(0.5))
        neg = K.relu(-inputs)
        neg = -self.beta * (K.sigmoid(neg) - K.cast_to_floatx(0.5))
        return pos + neg

    def get_config(self):
        config = {'alpha': float(self.alpha), 'beta': float(self.beta)}
        base_config = super(PSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _vgg_conv_block(input_layer, conv_layer=Conv2D,
                    pooling_layer=AveragePooling2D,
                    kernel_size=(12, 12), pooling_size=(6, 6),
                    conv_layer_activation='relu'):
    """
    VGG-like highly customisable convolutional network
    Parameters
    ----------
    input_layer: keras.layers.Input
        The first input layer of the network to plug on top of the VGG-like conv blocks
    conv_layer: keras.layers.convolutional._Conv (default: Conv2D)
        The convolutional layer to plug in each block
    pooling_layer: keras.layers.pooling._Pooling (default: AveragePooling2D)
        The pooling layer to plug in each block
    kernel_size: tuple (default: (12, 12))
        The size of each convolutional kernel
    pooling_size: tuple (default: (6, 6))
        The size of each pooling mask
    conv_layer_activation: object or str (default: 'relu')
        The activation function to be used after each convolutional layer.
        This parameter can be either a string or a custom keras layer
        object. If a string, this must correspond to one of the
        default activation functions supported by Keras.
    Returns
    -------
        keras tensor of the last layer of the network
    Notes:
    ------
        Please make sure that `conv_layer` and `pooling_layer` have the same rank
        (i.e. both 2D or both 3D) as well as the len of corresponding
        `kernel_size` and `pooling_size`.
    """

    prefix = input_layer.name.split('_')[0]

    if isinstance(conv_layer_activation, str):
        conv_activation_function = conv_layer_activation
        stack_activation_layer = False
    elif issubclass(conv_layer_activation, Layer):
        conv_activation_function = 'linear'
        stack_activation_layer = True
    else:
        conv_activation_function = conv_layer_activation
        stack_activation_layer = False

    # Block 1
    x = conv_layer(32, kernel_size=kernel_size, activation=conv_activation_function,
                   padding='same', name='{}_block1_conv1'.format(prefix))(input_layer)

    if stack_activation_layer:
        x = conv_layer_activation()(x)

    x = conv_layer(32, kernel_size=kernel_size, activation=conv_activation_function,
                   padding='same', name='{}_block1_conv2'.format(prefix))(x)

    if stack_activation_layer:
        x = conv_layer_activation()(x)

    x = pooling_layer(pool_size=pooling_size, strides=(2, 2),
                      padding='same', name='{}_block1_pool'.format(prefix))(x)

    # Block 2
    x = conv_layer(64, kernel_size=kernel_size, activation=conv_activation_function,
                   padding='same', name='{}_block2_conv1'.format(prefix))(x)

    if stack_activation_layer:
        x = conv_layer_activation()(x)

    x = conv_layer(64, kernel_size=kernel_size, activation=conv_activation_function,
                   padding='same', name='{}_block2_conv2'.format(prefix))(x)

    if stack_activation_layer:
        x = conv_layer_activation()(x)

    x = pooling_layer(pool_size=pooling_size, strides=(2, 2),
                      padding='same', name='{}_block2_pool'.format(prefix))(x)

    # Block 3
    x = conv_layer(128, kernel_size=kernel_size, activation=conv_activation_function,
                   padding='same', name='{}_block3_conv2'.format(prefix))(x)

    if stack_activation_layer:
        x = conv_layer_activation()(x)

    x = pooling_layer(pool_size=pooling_size, strides=(2, 2),
                      padding='same', name='{}_block3_pool'.format(prefix))(x)
    x = Flatten(name='{}_flatten_1'.format(prefix))(x)
    return x


def _tz_topology(zt_layer, conv_layer, kernel_size, pooling_layer, pooling_size,
                 dense_layer_activation='relu', conv_layer_activation='relu'):
    """utility function to build tz-convnet network topology"""
    zt_branch = _vgg_conv_block(input_layer=zt_layer,
                                conv_layer=conv_layer, kernel_size=kernel_size,
                                conv_layer_activation=conv_layer_activation,
                                pooling_layer=pooling_layer, pooling_size=pooling_size, )
    x = Dense(512, activation=dense_layer_activation, name='fc-1')(zt_branch)
    x = Dense(512, activation=dense_layer_activation, name='fc-2')(x)
    return x


def TZ_updown_classification(num_classes, optimizer=DEFAULT_OPT,
                             conv_layer=Conv2D, pooling_layer=AveragePooling2D,
                             kernel_size=(12, 12), pooling_size=(6, 6)):
    """VGG inspired Convolutional Networks
    Parameters
    ----------
    num_classes : int
        Number of classes to predict
    optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
        Instance of Keras optimizer to attach to the resulting network
    Other Parameters
    ----------------
    These parameters are passed to `_vgg_conv_block` function
    conv_layer: keras.layers.convolutional._Conv (default: Conv2D)
        The convolutional layer to plug in each block
    pooling_layer: keras.layers.pooling._Pooling (default: AveragePooling2D)
        The pooling layer to plug in each block
    kernel_size: tuple (default: (12, 12))
        The size of each convolutional kernel
    pooling_size: tuple (default: (6, 6))
        The size of each pooling mask
    """

    tz_layer = Input(shape=TZ_SHAPE, name='tz_input')
    x = _tz_topology(tz_layer, conv_layer, kernel_size, pooling_layer, pooling_size)

    # prediction layer
    predictions = Dense(num_classes, activation='softmax', name='prediction')(x)

    # Model
    model = Model(inputs=tz_layer, outputs=predictions, name='tz_updown_classification')
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model


def TZnet_regression_cosz(optimizer=DEFAULT_OPT, compile_model=True):
    """VGG inspired Convolutional Networks for a Regression problem
    See Also
    --------
    TZnet : function for TZnet for classification settings
    """

    tz_layer = Input(shape=TZ_SHAPE, name='tz_input')

    zt_branch = _vgg_conv_block(input_layer=tz_layer, conv_layer_activation='tanh')
    x = Dense(512, activation='tanh', name='tz_fc-1')(zt_branch)
    x = Dense(512, activation='tanh', name='tz_fc-2')(x)

    # prediction layer
    predictions = Dense(1, activation=K.cos, name='tz_prediction')(x)

    # Model
    model = Model(inputs=tz_layer, outputs=predictions, name='tz_net_regression')
    if compile_model:
        model.compile(loss=mse, optimizer=optimizer)
    return model


def TXnet_regression_cosx(optimizer=DEFAULT_OPT, compile_model=True):
    """
        V1:
            Conv 32, 64, 128 - tanh + GlobalAveragePooling
            Dense 128, 64
        V2:
            Conv 32, 64, 128 - tanh + GlobalAveragePooling
            Dense 256, 256
        V3: (not tested yet)
            Conv 32, 64, 128 - relu + Flatten
            Dense 512, 512
        V4: cosz net
    """

    KERNEL_SIZE = (12, 12)
    POOLING_SIZE = (4, 4)
    STRIDES_STEP = (2, 2)
    CONV_ACTIVATION = 'tanh'
    #
    tx_layer = Input(shape=TX_SHAPE, name='tx_input')
    x = Conv2D(32, kernel_size=KERNEL_SIZE, activation=CONV_ACTIVATION,
               padding='same', name='tx_block1_conv1')(tx_layer)
    x = Conv2D(64, kernel_size=KERNEL_SIZE, activation=CONV_ACTIVATION,
               padding='same', name='tx_block1_conv2')(x)
    x = Conv2D(128, kernel_size=KERNEL_SIZE, activation=CONV_ACTIVATION,
               padding='same', name='tx_block1_conv3')(x)
    x = AveragePooling2D(pool_size=POOLING_SIZE, strides=STRIDES_STEP,
                         padding='same', name='tx_block3_pool')(x)
    x = GlobalAveragePooling2D(name='tx_global_average_pooling2d_1')(x)  # Flatten(name='flatten')(x)

    x = Dense(128, activation='tanh', name='tx_fc-1')(x)
    x = Dense(64, activation='tanh', name='tx_fc-2')(x)

    # prediction layer
    predictions = Dense(1, activation=K.cos, name='tx_prediction')(x)

    # tx_branch = _vgg_conv_block(input_layer=tx_layer, conv_layer_activation='tanh')
    # x = Dense(512, activation='tanh', name='fc-1')(tx_branch)
    # x = Dense(512, activation='tanh', name='fc-2')(x)

    # Model
    model = Model(inputs=tx_layer, outputs=predictions, name='tx_net_regression_dirx_v1')
    if compile_model:
        model.compile(loss=mse, optimizer=optimizer)
    return model


def TYnet_regression_cosy(optimizer=DEFAULT_OPT, compile_model=True):
    """
        V1:
            Conv 32, 64, 128 - tanh + GlobalAveragePooling
            Dense 128, 64
        V2:
            Conv 32, 64, 128 - tanh + GlobalAveragePooling
            Dense 256, 256
        V3: (not tested yet)
            Conv 32, 64, 128 - relu + Flatten
            Dense 512, 512
        V4: cosz net
    """

    KERNEL_SIZE = (12, 12)
    POOLING_SIZE = (4, 4)
    STRIDES_STEP = (2, 2)
    CONV_ACTIVATION = 'tanh'

    ty_layer = Input(shape=TY_SHAPE, name='ty_input')

    x = Conv2D(32, kernel_size=KERNEL_SIZE, activation=CONV_ACTIVATION,
               padding='same', name='ty_block1_conv1')(ty_layer)
    x = Conv2D(64, kernel_size=KERNEL_SIZE, activation=CONV_ACTIVATION,
               padding='same', name='ty_block1_conv2')(x)
    x = Conv2D(128, kernel_size=KERNEL_SIZE, activation=CONV_ACTIVATION,
               padding='same', name='ty_block1_conv3')(x)
    x = AveragePooling2D(pool_size=POOLING_SIZE, strides=STRIDES_STEP,
                         padding='same', name='ty_block3_pool')(x)
    x = GlobalAveragePooling2D(name='ty_global_average_pooling2d_1')(x)  # Flatten(name='flatten')(x)

    x = Dense(128, activation='tanh', name='ty_fc-1')(x)
    x = Dense(64, activation='tanh', name='ty_fc-2')(x)
    #

    # ty_branch = _vgg_conv_block(input_layer=ty_layer, conv_layer_activation='tanh')
    # x = Dense(512, activation='tanh', name='fc-1')(ty_branch)
    # x = Dense(512, activation='tanh', name='fc-2')(x)

    # prediction layer
    predictions = Dense(1, activation=K.cos, name='ty_prediction')(x)

    # Model
    model = Model(inputs=ty_layer, outputs=predictions, name='ty_net_regression_diry_v1')
    if compile_model:
        model.compile(loss=mse, optimizer=optimizer)
    return model


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.maximum(K.sum(K.square(y_true - y_pred), axis=1, keepdims=True), K.epsilon()))


def distance_loss(y_true, y_pred):
    # 1.5707964 == np.pi/2
    t_dot = K.batch_dot(y_true, y_pred, axes=-1)  # K.sum(y_true * y_pred, axis=-1)
    t_dot = tf.where(K.greater(t_dot, 1.), K.ones_like(t_dot) * K.epsilon(), t_dot)
    t_dot = tf.where(K.less(t_dot, -1.), K.ones_like(t_dot) * K.epsilon(), t_dot)
    return tf.acos(t_dot)
    # return tf.where(tf.is_nan(acos), tf.ones_like(acos) * 1.5707964, acos)


def sum_of_square_f(tensors):
    tx, ty, tz = tensors
    return K.square(tx)+K.square(ty)+K.square(tz)


def DirectionNet(optimizer=DEFAULT_OPT, loss_weights=(1, 1, 1, 1, 1)):
    """Multi-input Multi-Output Network model for
    Director Cosines Estimation
    """

    dirx_model = TXnet_regression_cosx(compile_model=False)
    diry_model = TYnet_regression_cosy(compile_model=False)
    dirz_model = TZnet_regression_cosz(compile_model=False)

    tx_input, tx_output = dirx_model.input, dirx_model.output
    ty_input, ty_output = diry_model.input, diry_model.output
    tz_input, tz_output = dirz_model.input, dirz_model.output

    gather = concatenate([tx_output, ty_output, tz_output], name='direction_vector')
    sum_of_squares = Lambda(sum_of_square_f, name='sum_of_squares')([tx_output, ty_output, tz_output])

    direction_net = Model(inputs=[tx_input, ty_input, tz_input],
                          outputs=[tx_output, ty_output, tz_output, gather, sum_of_squares],
                          name='DirectionNet')
    direction_net.compile(optimizer=optimizer, loss=['mse', 'mse', 'mse', distance_loss, 'mse'],
                          loss_weights=list(loss_weights))
    return direction_net


def DirectionNetShared(optimizer=DEFAULT_OPT, loss_weights=(1, 1, 1, 1, 1)):
    """
    """
    def build_shared_layers(kernel_size=(12, 12), activation_fn='tanh', pool_size=(6, 6)):
        """"""
        # Block 1
        blk1_c1 = Conv2D(32, kernel_size=kernel_size, activation=activation_fn,
                         padding='same', name='block1_conv1')

        blk1_c2 = Conv2D(32, kernel_size=kernel_size, activation=activation_fn,
                         padding='same', name='block1_conv2')

        blk1_avg = AveragePooling2D(pool_size=pool_size, strides=(2, 2),
                                    padding='same', name='block1_pool')

        block1 = [blk1_c1, blk1_c2, blk1_avg]

        # Block 2
        blk2_c1 = Conv2D(64, kernel_size=kernel_size, activation=activation_fn,
                         padding='same', name='block2_conv1')

        blk2_c2 = Conv2D(64, kernel_size=kernel_size, activation=activation_fn,
                         padding='same', name='block2_conv2')

        blk2_avg = AveragePooling2D(pool_size=pool_size, strides=(2, 2),
                                    padding='same', name='block2_pool')

        block2 = [blk2_c1, blk2_c2, blk2_avg]

        # Block 3
        blk3_c1 = Conv2D(128, kernel_size=kernel_size, activation=activation_fn,
                         padding='same', name='block3_conv1')

        blk3_c2 = Conv2D(128, kernel_size=kernel_size, activation=activation_fn,
                         padding='same', name='block3_conv2')

        blk3_avg = AveragePooling2D(pool_size=pool_size, strides=(2, 2),
                                    padding='same', name='block3_pool')

        block3 = [blk3_c1, blk3_c2, blk3_avg]

        # Flatten
        flatten = Flatten(name='flatten')

        return block1, block2, block3, [flatten]

    def build_network(shared_layers, input_layer):
        """"""
        x = input_layer
        for block in shared_layers:
            for layer in block:
                x = layer(x)
        return x

    tx_layer = Input(shape=TX_SHAPE, name='tx_input')
    ty_layer = Input(shape=TY_SHAPE, name='ty_input')
    tz_layer = Input(shape=TZ_SHAPE, name='tz_input')

    shared_layers = build_shared_layers()

    x_tensors = build_network(shared_layers, tx_layer)
    y_tensors = build_network(shared_layers, ty_layer)
    z_tensors = build_network(shared_layers, tz_layer)

    # Tx Branch
    fc_x = Dense(512, activation='tanh', name='fc_x-1')(x_tensors)
    fc_x = Dense(512, activation='tanh', name='fc_x-2')(fc_x)
    cosx = Dense(1, activation=K.cos, name='cosx')(fc_x)

    # Ty Branch
    fc_y = Dense(512, activation='tanh', name='fc_y-1')(y_tensors)
    fc_y = Dense(512, activation='tanh', name='fc_y-2')(fc_y)
    cosy = Dense(1, activation=K.cos, name='cosy')(fc_y)

    # Tz Branch
    fc_z = Dense(512, activation='tanh', name='fc_z-1')(z_tensors)
    fc_z = Dense(512, activation='tanh', name='fc_z-2')(fc_z)
    cosz = Dense(1, activation=K.cos, name='cosz')(fc_z)

    direction_layer = concatenate([cosx, cosy, cosz], name='direction_vector')
    sum_of_squares = Lambda(sum_of_square_f, name='sum_of_squares')([cosx, cosy, cosz])

    direction_net = Model(inputs=[tx_layer, ty_layer, tz_layer],
                          outputs=[cosx, cosy, cosz, direction_layer, sum_of_squares],
                          name='direction_net_shared_layers')
    direction_net.compile(optimizer=optimizer,
                          loss=['mse', 'mse', 'mse', 'cosine_proximity', 'mse'],
                          loss_weights=list(loss_weights))
    return direction_net



# def TZXY_regression_energy_old(optimizer=DEFAULT_OPT):
#     """VGG inspired Convolutional Networks with multiple inputs
#
#     Parameters
#     ----------
#     optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
#         Instance of Keras optimizer to attach to the resulting network
#     """
#
#     tz_layer = Input(shape=TZ_SHAPE, name='tz_input')
#     xy_layer = Input(shape=XY_SHAPE, name='xy_input')
#
#     # Conv2D((12, 12)) - AveragePooling((6, 6))
#     tz_branch = _vgg_conv_block(input_layer=tz_layer,
#                                 conv_layer_activation=PReLU)
#
#     # Conv2D((12, 12)) - AveragePooling((6, 6))
#     xy_branch = _vgg_conv_block(input_layer=xy_layer,
#                                 conv_layer_activation=PReLU)
#     # top
#     x = concatenate([tz_branch, xy_branch], name='merge_concat')
#     x = Dense(1024, activation='linear', name='fc-1')(x)
#     x = PTanh()(x)
#     x = Dense(512, activation='linear', name='fc-2')(x)
#     x = PTanh()(x)
#
#     # def log10(x):
#     #     num = tf.log(x)
#     #     den = np.log(10)
#     #     log10_tens = num / den
#     #     log10_tens = tf.where(tf.is_nan(log10_tens), tf.ones_like(log10_tens), log10_tens)
#     #     return log10_tens
#
#     # prediction layer
#     prediction = Dense(1, activation='linear', name='prediction')(x)
#     prediction = PTanh()(prediction)
#
#     model = Model(inputs=[tz_layer, xy_layer], outputs=prediction, name='net_logE_prelu_ptanh')
#     model.compile(loss=mse, optimizer=optimizer)
#     return model


def _tz_branch_shallow(input_layer):
    prefix = input_layer.name.split('_')[0]

    kernel_size = (12, 12)
    pooling_size = (6, 6)
    strides_step = (2, 2)
    conv_act = 'relu'

    # Block 1
    x = Conv2D(32, kernel_size=kernel_size, activation=conv_act,
               padding='same', name='{}_block1_conv1'.format(prefix))(input_layer)
    x = Conv2D(32, kernel_size=kernel_size, activation=conv_act,
               padding='same', name='{}_block1_conv2'.format(prefix))(x)
    x = AveragePooling2D(pool_size=pooling_size, strides=strides_step,
                         padding='same', name='{}_block1_pool'.format(prefix))(x)

    # Block 2
    x = Conv2D(64, kernel_size=kernel_size, activation=conv_act,
               padding='same', name='{}_block2_conv1'.format(prefix))(x)
    x = Conv2D(64, kernel_size=kernel_size, activation=conv_act,
               padding='same', name='{}_block2_conv2'.format(prefix))(x)
    x = AveragePooling2D(pool_size=pooling_size, strides=strides_step,
                         padding='same', name='{}_block2_pool'.format(prefix))(x)

    # Block 3
    x = Conv2D(128, kernel_size=kernel_size, activation=conv_act,
               padding='same', name='{}_block3_conv2'.format(prefix))(x)
    x = AveragePooling2D(pool_size=pooling_size, strides=strides_step,
                         padding='same', name='{}_block3_pool'.format(prefix))(x)

    x = GlobalAveragePooling2D()(x)
    return x


def _xy_branch_shallow(input_layer):
    prefix = input_layer.name.split('_')[0]

    kernel_size = (12, 12)
    pooling_size = (4, 4)
    strides_step = (2, 2)
    conv_act = 'relu'

    # Block 1
    x = Conv2D(32, kernel_size=kernel_size, activation=conv_act,
               padding='same', name='{}_block1_conv1'.format(prefix))(input_layer)
    x = Conv2D(64, kernel_size=kernel_size, activation=conv_act,
               padding='same', name='{}_block1_conv2'.format(prefix))(x)
    x = Conv2D(128, kernel_size=kernel_size, activation=conv_act,
               padding='same', name='{}_block1_conv3'.format(prefix))(x)
    x = AveragePooling2D(pool_size=pooling_size, strides=strides_step,
                         padding='same', name='{}_block3_pool'.format(prefix))(x)

    x = GlobalAveragePooling2D()(x)
    return x


def TZXY_regression_logE_relu_psigmoid(optimizer=DEFAULT_OPT, alpha=8.0, beta=4.0):
    """VGG inspired Convolutional Networks with multiple inputs
    Parameters
    ----------
    optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
        Instance of Keras optimizer to attach to the resulting network
    alpha: float (default 8.0)
        Alpha value for the PSigmoid activation layer
    beta: float (default 4.0)
        Beta value for the PSigmoid activation layer
    """

    tz_layer = Input(shape=TZ_SHAPE, name='tz_input')
    xy_layer = Input(shape=XY_SHAPE, name='xy_input')

    # Conv2D((12, 12)) - AveragePooling((6, 6))
    tz_branch = _tz_branch_shallow(input_layer=tz_layer)

    # Conv2D((12, 12)) - AveragePooling((4, 4))
    xy_branch = _xy_branch_shallow(input_layer=xy_layer)

    # top
    x = concatenate([tz_branch, xy_branch], name='merge_concat')
    x = Dense(128, activation='linear', name='fc-1')(x)
    x = PSigmoid(alpha=alpha, beta=beta)(x)
    x = Dense(64, activation='linear', name='fc-2')(x)
    x = PSigmoid(alpha=alpha, beta=beta)(x)
    x = Dense(32, activation='linear', name='fc-3')(x)
    x = PSigmoid(alpha=alpha, beta=beta)(x)
    x = Dense(16, activation='linear', name='fc-4')(x)
    x = PSigmoid(alpha=alpha, beta=beta)(x)

    # prediction layer
    prediction = Dense(1, activation='linear', name='prediction')(x)

    model = Model(inputs=[tz_layer, xy_layer], outputs=prediction, name='net_logE_relu_only_psigmoid_linear')
    model.compile(loss=mse, optimizer=optimizer)
    return model


def TZXY_regression_logE_relu_tanh(optimizer=DEFAULT_OPT):
    """VGG inspired Convolutional Networks with multiple inputs
    Parameters
    ----------
    optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
        Instance of Keras optimizer to attach to the resulting network
    """

    tz_layer = Input(shape=TZ_SHAPE, name='tz_input')
    xy_layer = Input(shape=XY_SHAPE, name='xy_input')

    # Conv2D((12, 12)) - AveragePooling((6, 6))
    tz_branch = _tz_branch_shallow(input_layer=tz_layer)

    # Conv2D((12, 12)) - AveragePooling((4, 4))
    xy_branch = _xy_branch_shallow(input_layer=xy_layer)

    # top
    x = concatenate([tz_branch, xy_branch], name='merge_concat')
    x = Dense(128, activation='tanh', name='fc-1')(x)
    x = Dense(64, activation='tanh', name='fc-2')(x)
    x = Dense(32, activation='tanh', name='fc-3')(x)
    x = Dense(16, activation='tanh', name='fc-4')(x)
    # prediction layer
    prediction = Dense(1, activation='linear', name='prediction')(x)

    model = Model(inputs=[tz_layer, xy_layer], outputs=prediction, name='net_logE_shallow_relu_tanh_linear')
    model.compile(loss=mse, optimizer=optimizer)
    return model


def _tz_branch_vgg(input_layer):
    prefix = input_layer.name.split('_')[0]
    kernel_size = (12, 12)
    pooling_size = (6, 6)
    pooling_stride = (2, 2)

    # Block 1
    x = Conv2D(64, kernel_size, activation='relu', padding='same', name='{}_block1_conv1'.format(prefix))(input_layer)
    x = Conv2D(64, kernel_size, activation='relu', padding='same', name='{}_block1_conv2'.format(prefix))(x)
    x = MaxPooling2D(pooling_size, strides=pooling_stride,
                     name='{}_block1_pool'.format(prefix))(x)

    # Block 2
    x = Conv2D(128, kernel_size, activation='relu', padding='same', name='{}_block2_conv1'.format(prefix))(x)
    x = Conv2D(128, kernel_size, activation='relu', padding='same', name='{}_block2_conv2'.format(prefix))(x)
    x = MaxPooling2D(pooling_size, strides=pooling_stride, name='{}_block2_pool'.format(prefix))(x)

    # Block 3
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block3_conv1'.format(prefix))(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block3_conv2'.format(prefix))(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block3_conv3'.format(prefix))(x)
    x = MaxPooling2D(pooling_size, strides=pooling_stride, padding='same',
                     name='{}_block3_pool'.format(prefix))(x)

    # Block 4
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block4_conv1'.format(prefix))(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block4_conv2'.format(prefix))(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block4_conv3'.format(prefix))(x)
    x = MaxPooling2D(pooling_size, strides=pooling_stride, padding='same',
                     name='{}_block4_pool'.format(prefix))(x)

    x = GlobalAveragePooling2D(name='{}_global_average_pooling'.format(prefix))(x)
    return x


def _xy_branch_vgg(input_layer):
    prefix = input_layer.name.split('_')[0]
    kernel_size = (12, 12)
    pooling_size = (4, 4)
    pooling_stride = (2, 2)

    # Block 1
    x = Conv2D(64, kernel_size, activation='relu', padding='same', name='{}_block1_conv1'.format(prefix))(input_layer)
    x = Conv2D(64, kernel_size, activation='relu', padding='same', name='{}_block1_conv2'.format(prefix))(x)

    x = MaxPooling2D(pooling_size, strides=pooling_stride, name='{}_block1_pool'.format(prefix), padding='same')(x)

    # Block 2
    x = Conv2D(128, kernel_size, activation='relu', padding='same', name='{}_block2_conv1'.format(prefix))(x)
    x = Conv2D(128, kernel_size, activation='relu', padding='same', name='{}_block2_conv2'.format(prefix))(x)
    x = MaxPooling2D(pooling_size, strides=pooling_stride, name='{}_block2_pool'.format(prefix), padding='same')(x)

    # Block 3
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block3_conv1'.format(prefix))(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block3_conv2'.format(prefix))(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='{}_block3_conv3'.format(prefix))(x)
    x = MaxPooling2D(pooling_size, strides=pooling_stride, name='{}_block3_pool'.format(prefix), padding='same')(x)

    x = GlobalAveragePooling2D(name='{}_global_average_pooling'.format(prefix))(x)
    return x


def TZXY_regression_logE_vgg(optimizer=DEFAULT_OPT):
    """VGG inspired Convolutional Networks with multiple inputs
    Parameters
    ----------
    optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
        Instance of Keras optimizer to attach to the resulting network
    """

    tz_layer = Input(shape=TZ_SHAPE, name='tz_input')
    xy_layer = Input(shape=XY_SHAPE, name='xy_input')

    # Conv2D((12, 12)) - AveragePooling((6, 6))
    tz_branch = _tz_branch_vgg(input_layer=tz_layer)

    # Conv2D((12, 12)) - AveragePooling((6, 6))
    xy_branch = _xy_branch_vgg(input_layer=xy_layer)

    # top
    x = add([tz_branch, xy_branch], name='merge_add')
    x = Dense(256, activation='tanh', name='fc-1-256')(x)
    x = Dropout(rate=.5, name='dropout-256-.5')(x)
    x = Dense(128, activation='tanh', name='fc-2-128')(x)
    x = Dropout(rate=.5, name='dropout-128-.5')(x)
    x = Dense(64, activation='tanh', name='fc-3-64')(x)
    x = Dropout(rate=.5, name='dropout-64-.3')(x)
    x = Dense(32, activation='tanh', name='fc-4-32')(x)
    x = Dropout(rate=.5, name='dropout-32-.3')(x)
    x = Dense(16, activation='tanh', name='fc-5-16')(x)

    # prediction layer
    prediction = Dense(1, activation='linear', name='prediction')(x)

    model = Model(inputs=[tz_layer, xy_layer], outputs=prediction,
                  name='net_logE_vgg_large_kernel_tanh_add_fc')
    model.compile(loss=mse, optimizer=optimizer)
    return model


def _tz_branch_resnet(input_tensor):
    prefix = input_tensor.name.split('_')[0]

    x = conv_block(input_tensor, kernel_size=6, filters=[32, 32, 128], stage=2,
                   block='{}_a'.format(prefix))
    x = identity_block(x, kernel_size=3, filters=[64, 64, 128], stage=2,
                       block='{}_b'.format(prefix))
    x = AveragePooling2D((6, 6), name='avg_pool_{}'.format(prefix), padding='same')(x)

    # GlobalAveragePooling2D(name='{}_global_average_pooling'.format(prefix))(x)
    x = Flatten(name='{}_flatten'.format(prefix))(x)

    return x


def _xy_branch_resnet(input_tensor):
    """"""
    prefix = input_tensor.name.split('_')[0]

    x = conv_block(input_tensor, kernel_size=6, filters=[32, 64, 128], stage=2,
                   block='{}_a'.format(prefix))
    x = AveragePooling2D((4, 4), name='avg_pool_{}'.format(prefix), padding='same')(x)

    # GlobalAveragePooling2D(name='{}_global_average_pooling'.format(prefix))(x)
    x = Flatten(name='{}_flatten'.format(prefix))(x)

    return x


def TZXY_regression_logE_residual(optimizer=DEFAULT_OPT):
    """ResNet inspired Convolutional Networks with multiple inputs
    predicting (in regression) value for logE
    Parameters
    ----------
    optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
        Instance of Keras optimizer to attach to the resulting network
    """

    tz_layer = Input(shape=TZ_SHAPE, name='tz_input')
    xy_layer = Input(shape=XY_SHAPE, name='xy_input')

    # TZ Branch
    tz_branch = _tz_branch_resnet(tz_layer)
    xy_branch = _xy_branch_resnet(xy_layer)

    # top
    x = concatenate([tz_branch, xy_branch], name='merge_concat')
    x = Dense(512, activation='tanh', name='fc-1-512')(x)
    x = Dense(64, activation='tanh', name='fc-2-64')(x)
    x = Dense(16, activation='tanh', name='fc-3-16')(x)

    # prediction layer
    prediction = Dense(1, activation='linear', name='prediction')(x)

    model = Model(inputs=[tz_layer, xy_layer], outputs=prediction,
                  name='net_logE_residual_shallow')
    model.compile(loss=mse, optimizer=optimizer)
    return model


def TXYZnet(num_classes, optimizer=DEFAULT_OPT):
    """VGG inspired Convolutional Networks with multiple inputs
    Parameters
    ----------
    num_classes : int
        Number of classes to predict
    optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
        Instance of Keras optimizer to attach to the resulting network
    """

    txyz_layer = Input(shape=TXYZ_SHAPE_TIME_DISTRIBUTED, name='txyz_input')

    prefix = txyz_layer.name.split('_')[0]
    kernel_size_3d = (12, 12, 12)
    pooling_size_3d = (6, 6, 6)
    kernel_size_2d = (12, 12)
    pooling_size_2d = (6, 6)

    if K.image_data_format() == 'channels_last':
        sum_axis = (2, 3)
    else:  # channels_first
        sum_axis = (3, 4)

    # Block 1
    x = TimeDistributed(Conv3D(32, kernel_size=kernel_size_3d, activation='relu',
                               padding='same', name='{}_block1_conv1'.format(prefix)),
                        name='td_{}_block1_conv1'.format(prefix))(txyz_layer)
    x = TimeDistributed(Conv3D(32, kernel_size=kernel_size_3d, activation='relu',
                               padding='same', name='{}_block1_conv2'.format(prefix)),
                        name='td_{}_block1_conv2'.format(prefix))(x)
    x = TimeDistributed(AveragePooling3D(pool_size=pooling_size_3d,
                                         strides=(2, 2, 2), padding='same',
                                         name='{}_block1_pool'.format(prefix)),
                        name='td_{}_block1_pool'.format(prefix))(x)
    # Block 2
    # x = TimeDistributed(Conv3D(64, kernel_size=kernel_size_3d, activation='relu',
    #                            padding='same', name='{}_block2_conv1'.format(prefix)),
    #                     name='td_{}_block2_conv1'.format(prefix))(x)
    # x = TimeDistributed(Conv3D(64, kernel_size=kernel_size_3d, activation='relu',
    #                            padding='same', name='{}_block2_conv2'.format(prefix)),
    #                     name='td_{}_block2_conv2'.format(prefix))(x)
    # operational Layer

    x = Lambda(lambda t: K.sum(t, axis=sum_axis), name='xy_squashing')(x)
    if K.image_data_format() == "channels_first":
        x = Permute((2, 1, 3))(x)
    prefix = 'tz'
    x = Conv2D(64, kernel_size=kernel_size_2d, activation='relu',
               padding='same', name='{}_block3_conv1'.format(prefix))(x)
    x = Conv2D(64, kernel_size=kernel_size_2d, activation='relu',
               padding='same', name='{}_block3_conv2'.format(prefix))(x)
    x = AveragePooling2D(pool_size=pooling_size_2d, strides=(2, 2),
                         padding='same', name='{}_block3_pool'.format(prefix))(x)
    # Block 3
    x = Conv2D(128, kernel_size=kernel_size_2d, activation='relu',
               padding='same', name='{}_block4_conv2'.format(prefix))(x)
    x = GlobalAveragePooling2D(name='{}_global_pool'.format(prefix))(x)
    # top
    x = Dense(128, activation='relu', name='fc-1')(x)

    # prediction layer
    predictions = Dense(num_classes, activation='softmax', name='prediction')(x)

    model = Model(inputs=txyz_layer, outputs=predictions, name='txyz_net')
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model


def TZXY_numu_nue_classification(num_classes, optimizer=DEFAULT_OPT):
    """VGG inspired Convolutional Networks
    Parameters
    ----------
    num_classes : int
        Number of classes to predict
    optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
        Instance of Keras optimizer to attach to the resulting network
    V1: xy and tz branch identical
        Concat + Dense (256)
    V2: xy branch :
            (12, 12) - (3, 3)
        tz branch :
            (12, 12) - (6, 6)
        Concat
        Dense 512, 512
    """

    tz_layer = Input(shape=TZ_SHAPE, name='tz_input')
    xy_layer = Input(shape=XY_SHAPE, name='xy_input')

    tz_branch = _vgg_conv_block(tz_layer)
    xy_branch = _vgg_conv_block(xy_layer)  # , pooling_size=(3, 3))

    x = concatenate([tz_branch, xy_branch], name="merge_concat")
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    # prediction layer
    predictions = Dense(num_classes, activation='softmax', name='prediction')(x)

    # Model
    model = Model(inputs=[tz_layer, xy_layer], outputs=predictions, name='tzxy_numu_nue_net_v1')
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model


def leaf_classification(num_classes, optimizer=DEFAULT_OPT,
                             conv_layer=Conv2D, pooling_layer=AveragePooling2D,
                             kernel_size=(3, 3), pooling_size=(3, 3)):
    """VGG inspired Convolutional Networks
    Parameters
    ----------
    num_classes : int
        Number of classes to predict
    optimizer : keras.optimizers.Optimizer (default: Adadelta() - with def params)
        Instance of Keras optimizer to attach to the resulting network
    Other Parameters
    ----------------
    These parameters are passed to `_vgg_conv_block` function
    conv_layer: keras.layers.convolutional._Conv (default: Conv2D)
        The convolutional layer to plug in each block
    pooling_layer: keras.layers.pooling._Pooling (default: AveragePooling2D)
        The pooling layer to plug in each block
    kernel_size: tuple (default: (12, 12))
        The size of each convolutional kernel
    pooling_size: tuple (default: (6, 6))
        The size of each pooling mask
    """
    input_layer = Input(shape=LEAF_SHAPE, name='leaf_input')
    x = _tz_topology(input_layer, conv_layer, kernel_size, pooling_layer, pooling_size)

    # prediction layer
    predictions = Dense(num_classes, activation='softmax', name='prediction')(x)

    # Model
    model = Model(inputs=input_layer, outputs=predictions, name='leaf_position_classification')
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model


def train_neural_network(network_model, training_generator, steps_per_epoch,
                         validation_generator=None, validation_steps=None,
                         verbose=1, epochs=100, batch_size=64, class_weights=None, callbacks=None,
                         log_suffix='', checkpoint_folder='', save_best_only=True, no_stopping=False):
    """
    Parameters
    ----------
    network_model:  keras.models.Model
        Instance of the network to train
    training_generator:   Python generator
        generator function to pass to fit_generator method to generate **training** data
    steps_per_epoch: int
        number of steps the training generator has to run to declare one epoch finished
    validation_generator: Python generator
        generator function to pass to fit_generator to generate **validation** data
    validation_steps: int
        number of steps the validation generator has to run in each validation step
    verbose: int (default 1)
        Verbosity of fit process. By default, verbosity is ON
    epochs: int (default 100)
        number of training epochs
    batch_size: int (default 64)
        size of the batch
    class_weights: dict (default None)
        Dictionary mapping a weight for each class
    callbacks: list (default None)
        List of keras.callbacks objects to plug during the training process
        By default, EarlyStopping and ModelCheckPoint are always plugged in.
    log_suffix: str (default '')
        suffix string to be used when saving model checkpoint log
    checkpoint_folder: str (default '')
        Path to the folder in which ModelCheckpoint file(s)
        will be saved. If no path is provided, the default checkpoint folder
        will be used (see network_models.CHECKPOINT_FOLDER_PATH)
    save_best_only: bool (default True)
        Flag to be fed into the ModelCheckpoint Keras Callback
        indicating whether saving ONLY best model or saving all the chekpoints.
    Returns
    -------
        network training history object (as returned by `fit_generator`)
    """

    if checkpoint_folder:
        checkpoint_folder_path = checkpoint_folder
    else:
        checkpoint_folder_path = CHECKPOINT_FOLDER_PATH

    if not p.exists(checkpoint_folder_path):
        makedirs(checkpoint_folder_path)

    checkpoint_fname = p.join(checkpoint_folder_path, "{}_{}_{}_{}.hdf5".format(network_model.name,
                                                                                batch_size, epochs, log_suffix))
    default_callbacks = [ModelCheckpoint(checkpoint_fname, save_best_only=save_best_only),]
    if not no_stopping:
        default_callbacks.append(EarlyStopping(monitor="val_loss", patience=5),)

    if callbacks is None:
        callbacks = default_callbacks
    else:
        callbacks = callbacks + default_callbacks

    net_history = network_model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                              validation_data=validation_generator, validation_steps=validation_steps,
                                              verbose=verbose, class_weight=class_weights, callbacks=callbacks)
    return net_history


def inference_step(network_model, test_data_generator, predict_steps,
                   metadata_generator, categorical=True):
    """
    Parameters
    ----------
    network_model:  keras.models.Model
        Instance of the network to be used in inference mode for predictions
    test_data_generator: Python generator
        generator function to generate **test** data to predict and corresponding expected value
    predict_steps: int
        Number of steps to run the `test_data_generator`
    metadata_generator: Python generator
        generator function to generate metadata according to corresponding test data
    categorical: bool (default True)
        Whether the prediction of the network model will be categorical (classification, tipically)
        or not. Default is True - thus expected to be used for supervised classification learning settings.
    Returns
    -------
        metadata: pandas.DataFrame
            Pandas dataframe with all the metadata generated by the `metadata` generator during
            the inference step
        y_true : array-like shape=[test_samples, ]
            Array of the ground truth for test data
        y_pred : array-like shape=[test_samples, ]
            Array of predictions of the network for each test sample
        Y_probs : array-like [test_samples, n_classes]
            Matrix of predictions (probability, tipically) as generated by
            `network_model.predict`.
    Notes
    -----
    `Y_probs` is returned **only** if `categorical` parameter is set to True.
    """

    y_true = list()
    y_pred = list()
    Y_probs = None
    metadata = None

    for _ in tqdm(range(predict_steps)):
        X_batch, Y_batch_true = next(test_data_generator)
        metadata_batch = next(metadata_generator)
        # Save Metadata
        if metadata is None:
            metadata = metadata_batch
        else:
            metadata = pd.concat((metadata, metadata_batch), ignore_index=True)

        Y_batch_pred = network_model.predict_on_batch(X_batch)

        if categorical:
            if Y_probs is None:
                Y_probs = Y_batch_pred
            else:
                Y_probs = np.vstack((Y_probs, Y_batch_pred))

            y_true.append(np.argmax(Y_batch_true, axis=1))
            y_pred.append(np.argmax(Y_batch_pred, axis=1))
        else:
            y_true.append(Y_batch_true)
            y_pred.append(Y_batch_pred.ravel())

    y_true = np.hstack(np.asarray(y_true))
    y_pred = np.hstack(np.asarray(y_pred))

    if categorical:
        return metadata, y_true, y_pred, Y_probs
    return metadata, y_true, y_pred


