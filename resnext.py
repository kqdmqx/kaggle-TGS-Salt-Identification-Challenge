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
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate, Dot, Multiply
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


ACTIVATION = "relu"
def gen_conv_names(base):
    conv_name = base + '_conv'
    bn_name = base + "_bn"
    act_name = base + "_{}".format(ACTIVATION)
    return conv_name, bn_name, act_name

def convolution_block(base_name, x, filters, size, strides=(1,1), padding='same', activation=True):
    conv_name, bn_name, act_name = gen_conv_names(base_name)
    x = Conv2D(filters, size, strides=strides, padding=padding, name=conv_name)(x)
    x = BatchNormalization(name=bn_name)(x)
    if activation == True:
        x = Activation(ACTIVATION, name=act_name)(x)
    return x

def se_block(blockInput, bottle=4):
    channel_cnt = int(blockInput.shape[-1])
    
    x = GlobalMaxPooling2D(data_format="channels_last")(blockInput)
    x = Dense(int(channel_cnt / bottle))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(channel_cnt)(x)
    x = Activation("sigmoid")(x)
    x = Reshape((1, 1, channel_cnt))(x)
    x = Multiply()([blockInput, x])
    return x

def scse_block(blockInput, bottle=4, ssigmoid=False, alpha=0.0001):
    channel_cnt = int(blockInput.shape[-1])
    img_size = int(blockInput.shape[1])
    img_size2 = img_size * img_size

    x = GlobalMaxPooling2D(data_format="channels_last")(blockInput)
    x = Dense(int(channel_cnt / bottle))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(channel_cnt, kernel_regularizer=l2(alpha))(x)
    x = Activation("sigmoid")(x)
    x = Reshape((1, 1, channel_cnt))(x)
    x = Multiply()([blockInput, x])
    

    y = Conv2D(1, (1, 1), padding="same")(blockInput)
    if ssigmoid:
        y = Activation("elu")(y)
    if img_size >= 50:
        y_pooling = (4, 4) if img_size % 2 == 1 else (2, 2)
        kernel_size = (5, 5) if img_size % 2 == 1 else (3, 3)
        padding = "valid" if img_size % 2 == 1 else "same"
        
        y = MaxPooling2D(y_pooling)(y)

        pool_size = int(y.shape[1])
        pool_size2 = pool_size * pool_size

        y = Reshape((pool_size2, ))(y)
        y = Dense(int(pool_size2 / bottle))(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Dense(pool_size2, kernel_regularizer=l2(alpha))(y)
        y = Activation("sigmoid")(y)
        y = Reshape((pool_size, pool_size, 1))(y)
        y = Conv2DTranspose(1, kernel_size, strides=y_pooling, padding=padding)(y)
    else:
        y = Reshape((img_size2, ))(y)
        y = Dense(int(img_size2 / bottle))(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Dense(img_size2, kernel_regularizer=l2(alpha))(y)
        y = Activation("sigmoid")(y)
        y = Reshape((img_size, img_size, 1))(y)

    y = Multiply()([blockInput, y])

    z = Add()([x, y])
    return z


def residual_block(base_name, blockInput, num_filters=16, use_se=False, ssigmoid=False, se_l2_alpha=0.0001):
    x = Activation(ACTIVATION, name=base_name + "_{}".format(ACTIVATION))(blockInput)
    x = BatchNormalization(name=base_name + "_bn")(x)
    x = convolution_block(base_name + "_1", x, num_filters, (3,3) )
    x = convolution_block(base_name + "_2", x, num_filters, (3,3), activation=False)
    if use_se:
        x = scse_block(x, ssigmoid=ssigmoid, alpha=se_l2_alpha)
    x = Add(name=base_name + "_add")([blockInput, x])
    return x


def bottle_neck_block(base_name, x, filters, size, height=4, strides=(1,1), padding='same', activation=True):
    conv_name, bn_name, act_name = gen_conv_names(base_name)
    input_channel = int(x.shape[-1])
    x = Conv2D(height, (1, 1), strides=strides, padding=padding)(x)
    x = Conv2D(filters, size, strides=strides, padding=padding, name=conv_name)(x)
    x = Conv2D(input_channel, (1, 1), strides=(1, 1), padding=padding)(x)
    x = BatchNormalization(name=bn_name)(x)
    if activation:
        x = Activation(Activation)(x)
    return x


def residual_block_next(base_name, blockInput, num_filters=16, height=4, use_se=False, ssigmoid=False, se_l2_alpha=0.0001):
    input_channel = int(blockInput.shape[-1])

    group_num = input_channel // height

    parts = []

    for i in range(group_num):
        parts.append(bottle_neck_block(base_name, blockInput, num_filters, (3, 3), activation=False)(blockInput))
    x = Add()(parts + [blockInput])

    
    return x


# Build model
def _build_model(input_layer, start_neurons, DropoutRatio=0.5, use_se=False, use_sigmoid=True, ssigmoid=False, se_l2_alpha=0.00001):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = residual_block("en1_b", conv1, start_neurons * 1, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    conv1 = residual_block("en1_c", conv1, start_neurons * 1, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = residual_block("en2_b", conv2, start_neurons * 2, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    conv2 = residual_block("en2_c", conv2, start_neurons * 2, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = residual_block("en3_b", conv3, start_neurons * 4, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    conv3 = residual_block("en3_c", conv3, start_neurons * 4, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = residual_block("en4_b", conv4, start_neurons * 8, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    conv4 = residual_block("en4_c", conv4, start_neurons * 8, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = residual_block("vm_b", convm, start_neurons * 16, use_se=False)
    convm = residual_block("vm_c", convm, start_neurons * 16, use_se=False)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = residual_block("de4_b", uconv4, start_neurons * 8, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    uconv4 = residual_block("de4_c", uconv4, start_neurons * 8, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    uconv4 = Activation(ACTIVATION)(uconv4)
    
    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = residual_block("de3_b", uconv3, start_neurons * 4, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    uconv3 = residual_block("de3_c", uconv3, start_neurons * 4, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = residual_block("de2_b", uconv2, start_neurons * 2, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    uconv2 = residual_block("de2_c", uconv2, start_neurons * 2, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    uconv2 = Activation(ACTIVATION)(uconv2)
    
    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = residual_block("de1_b", uconv1, start_neurons * 1, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    uconv1 = residual_block("de1_c", uconv1, start_neurons * 1, use_se=use_se, ssigmoid=ssigmoid, se_l2_alpha=se_l2_alpha)
    uconv1 = Activation(ACTIVATION)(uconv1)
    
    uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)
    
    return output_layer

