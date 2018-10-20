
# ## import

# In[1]:

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

from tqdm import tqdm_notebook, tnrange, tqdm
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add, Dense, SpatialDropout2D
from keras.layers.core import Lambda, Reshape, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate, Dot, Multiply, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import keras
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  # ,save_img
from keras.callbacks import CSVLogger


from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.regularizers import l2
from keras.losses import binary_crossentropy

import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

print(keras.__version__)
print(tf.__version__)

import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

# from https://github.com/neptune-ml/open-solution-salt-identification
from common_blocks.augmentation import iaa, PadFixed, RandomCropFixedSize
from common_blocks.utils import plot_list

import gc
