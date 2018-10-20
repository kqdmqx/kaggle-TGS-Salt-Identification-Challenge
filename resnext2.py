import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

# import cv2
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add, Dense
from keras.layers.core import Lambda, Reshape, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate, Dot, Multiply, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import keras
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  # ,save_img
from keras.callbacks import CSVLogger


from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.regularizers import l2


import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

print(keras.__version__)
print(tf.__version__)


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def GroupConv2D(filters, kernel_size, conv_params, conv_name, strides=(1, 1), cardinality=32):

    def layer(input_tensor):
        grouped_channels = int(input_tensor.shape[-1]) // cardinality
        blocks = []
        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input_tensor)
            name = conv_name + '_' + str(c)
            x = Conv2D(grouped_channels, kernel_size, strides=strides,
                       name=name, **conv_params)(x)
            blocks.append(x)

        x = Concatenate(axis=-1)(blocks)
        return x
    return layer


def conv_block(filters, stage, block, strides=(2, 2)):
    """The conv block is the block that has conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        strides: tuple of integers, strides for conv (3x3) layer in block
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):

        # extracting params and names for layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = Activation('relu', name=relu_name)(input_tensor)
        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), conv_params, conv_name + '2', strides=strides)(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)

        shortcut = Conv2D(filters * 2, (1, 1), name=sc_name, strides=strides, **conv_params)(input_tensor)
        shortcut = BatchNormalization(name=sc_name +'_bn', **bn_params)(shortcut)
        x = Add()([x, shortcut])

        return x

    return layer


def identity_block(filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = Activation('relu', name=relu_name)(input_tensor)
        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), conv_params, conv_name + '2')(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)

        x = Add()([x, input_tensor])

        return x

    return layer


ACTIVATION = "relu"


def _build_model(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = identity_block(start_neurons * 1, 11, 2)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = identity_block(start_neurons * 2, 12, 2)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 2)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = identity_block(start_neurons * 8, 14, 2)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = identity_block(start_neurons * 16, 21, 0)(convm)
    convm = identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 2)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 2)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 2)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 2)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


def Scse(re=16):
    def layer(input_tensor):
        channel_cnt = int(input_tensor.shape[-1])
        x = GlobalMaxPooling2D(data_format="channels_last")(input_tensor)
        x = Dense(int(channel_cnt // re))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(channel_cnt)(x)
        x = Activation("sigmoid")(x)
        x = Reshape((1, 1, channel_cnt))(x)
        x = Multiply()([blockInput, x])

        y = Conv2D(1, (1, 1), padding="same")(input_tensor)
        y = Activation("sigmoid")(y)
        y = Multiply()([blockInput, y])

        z = Add()([x, y])
        return z

    return layer


def scse_block(filters, re=16, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = Activation('relu', name=relu_name)(input_tensor)
        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), conv_params, conv_name + '2')(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        # x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = Conv2D(filters, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Scse(re)(x)

        x = Add()([x, input_tensor])

        return x

    return layer




def _build_linknet_scse(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = scse_block(start_neurons * 1, 11, 2)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = scse_block(start_neurons * 2, 12, 2)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = scse_block(start_neurons * 4, 13, 2)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = scse_block(start_neurons * 8, 14, 2)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = identity_block(start_neurons * 16, 21, 0)(convm)
    convm = identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = Add()([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = scse_block(start_neurons * 8, 34, 2)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = Add()([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = scse_block(start_neurons * 4, 33, 2)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = Add()([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = scse_block(start_neurons * 2, 32, 2)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = Add()([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = scse_block(start_neurons * 1, 31, 2)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    # uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


def _build_unet_scse(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = scse_block(start_neurons * 1, 11, 2)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = scse_block(start_neurons * 2, 12, 2)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = scse_block(start_neurons * 4, 13, 2)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = scse_block(start_neurons * 8, 14, 2)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = identity_block(start_neurons * 16, 21, 0)(convm)
    convm = identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = scse_block(start_neurons * 8, 34, 2)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = scse_block(start_neurons * 4, 33, 2)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = scse_block(start_neurons * 2, 32, 2)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = scse_block(start_neurons * 1, 31, 2)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer