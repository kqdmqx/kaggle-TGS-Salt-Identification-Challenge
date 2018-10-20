from my_dependency import *
from my_metrics import *


# ## model structure --resnext & scse

# download blocks code from 
# 
# 1. ResNet https://github.com/qubvel/classification_models
# 2. Unet https://github.com/qubvel/segmentation_models

# ### resnext block

# In[77]:

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


def GroupConv2D(filters, kernel_size, conv_params, conv_name, strides=(1, 1), cardinality=8):

    def layer(input_tensor):
        print(conv_name, "-i-", input_tensor.shape)
        grouped_channels = int(input_tensor.shape[-1]) // cardinality
        blocks = []
        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input_tensor)
            name = conv_name + '_' + str(c)
            x = Conv2D(grouped_channels, kernel_size, strides=strides,
                       name=name, **conv_params)(x)
            blocks.append(x)

        x = Concatenate(axis=-1)(blocks)
        print(conv_name, "-o-", x.shape)
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

        x = BatchNormalization(name=bn_name + '0', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '0')(x)
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

        x = BatchNormalization(name=bn_name + '0', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '0')(x)
        x = Conv2D(filters // 2, (1, 1), name=conv_name + '1', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters // 2, (3, 3), conv_params, conv_name + '2')(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        # x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = Conv2D(filters, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Add()([x, input_tensor])

        return x

    return layer


def basic_identity_block(filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        x = Add()([x, input_tensor])
        return x

    return layer


def basic_conv_block(filters, stage, block, strides=(2, 2)):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        shortcut = x
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = Add()([x, shortcut])
        return x

    return layer



def basic_conv_block2(filters, stage, block, strides=(2, 2)):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        # x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        # x = Activation('relu', name=relu_name + '1')(x)
        # shortcut = x
        x = ZeroPadding2D(padding=(1, 1))(input_tensor)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(input_tensor)
        x = Add()([x, shortcut])
        return x

    return layer


# ### scse block

# In[62]:

def Scse(re=16):
    def layer(input_tensor):
        channel_cnt = int(input_tensor.shape[-1])

        x = GlobalAveragePooling2D(data_format="channels_last")(input_tensor)
        x = Dense(int(channel_cnt // re))(x)
        x = Activation("relu")(x)
        x = Dense(channel_cnt)(x)
        x = Activation("sigmoid")(x)
        x = Reshape((1, 1, channel_cnt))(x)
        x = Multiply()([input_tensor, x])

        y = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal")(input_tensor)
        y = Activation("sigmoid")(y)
        y = Multiply()([input_tensor, y])

        z = Add()([x, y])
        return z

    return layer


def scse_block(re, stage, block):
    def layer(input_tensor):
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
        x = BatchNormalization(name="scse_" + bn_name, **bn_params)(input_tensor)
        x = Activation('relu', name="scse_" + relu_name)(x)
        x = Scse(re)(x)
        return x
    return layer


def scse_block2(re, stage, block):
    def layer(input_tensor):
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
        x = BatchNormalization(name="scse_" + bn_name, **bn_params)(input_tensor)
        x = Scse(re)(x)
        x = Activation('relu', name="scse_" + relu_name)(x)
        return x
    return layer


def identity_scse_block(filters, stage, block, re=8):
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

        x = BatchNormalization(name=bn_name + '0', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name)(x)
        x = Conv2D(filters // 2, (1, 1), name=conv_name + '1', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters // 2, (3, 3), conv_params, conv_name + '2')(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        x = Conv2D(filters, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Scse(re)(x)

        x = Add()([x, input_tensor])

        return x

    return layer

def conv_scse_block(filters, stage, block, re=8):
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

        x = BatchNormalization(name=bn_name + '0', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name)(x)
        x = Conv2D(filters // 2, (1, 1), name=conv_name + '1', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters // 2, (3, 3), conv_params, conv_name + '2')(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        # x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = Conv2D(filters, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Scse(re)(x)
        x = BatchNormalization(name=bn_name + '4', **bn_params)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(input_tensor)
        shortcut = BatchNormalization(name=sc_name +'_bn', **bn_params)(shortcut)
        x = Add()([x, shortcut])

        return x

    return layer

def basic_identity_scse_block(filters, stage, block, re=8):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Scse(re)(x)

        x = Add()([x, input_tensor])
        return x

    return layer


def basic_conv_scse_block(filters, stage, block, strides=(2, 2), re=8):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        shortcut = x
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Scse(re)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = Add()([x, shortcut])
        return x

    return layer


# ### multi-task block

# In[13]:

def classifier_block(classes, middle, name):
    def layer(input_tensor):
        x = GlobalAveragePooling2D()(input_tensor)
        # x = Flatten()(x)
        print("clf --I--", x.shape)
        x = BatchNormalization()(x)
        x = Dense(middle)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(classes)(x)
        x = Activation("softmax", name=name)(x)
        print("clf --O--", x.shape)
        return x

    return layer


def fusion_block(figsize):
    def layer(input_tensor):
        # cons = tf.constant(value=[1], dtype=tf.float32, shape=(1, figsize, figsize, 1), name='fusion_cons', verify_shape=False)
        classes = int(input_tensor.shape[1])
        print("fusion --I--:", input_tensor.shape)
        x = Reshape((1, 1, classes))(input_tensor)
        # x = Multiply()([x, cons])
        x = UpSampling2D((figsize, figsize))(x)
        print("fusion --0--:", x.shape)
        return x
    return layer


# ### 3-tasks6 deep + scse +  conv + 3463

# In[90]:

ACTIVATION = "relu"
REDUCTION = 8


def _build_unet34magicbasic_multi6(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 128
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 2)(conv1)
    conv1 = scse_block2(REDUCTION, 11, 2)(conv1)

    # 128 -> 64
    # pool1 = MaxPooling2D((2, 2))(conv1)
    # pool1 = Dropout(DropoutRatio / 2)(pool1)
    # conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_conv_block2(start_neurons * 2, 12, 99)(conv1)

    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 2)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 34)(conv2)
    conv2 = scse_block2(REDUCTION, 12, 98)(conv2)

    # 64 -> 32
    # pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(DropoutRatio)(pool2)
    # conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_conv_block2(start_neurons * 4, 13, 99)(conv2)

    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 2)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 3)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 4)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 5)(conv3)
    conv3 = scse_block2(REDUCTION, 13, 98)(conv3)

    # 32 -> 16
    # pool3 = MaxPooling2D((2, 2))(conv3)
    # pool3 = Dropout(DropoutRatio)(pool3)
    # conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_conv_block2(start_neurons * 8, 14, 99)(conv3)

    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 2)(conv4)
    conv4 = scse_block2(REDUCTION, 14, 98)(conv4)

    # 16 -> 8
    # pool4 = MaxPooling2D((2, 2))(conv4)
    # pool4 = Dropout(DropoutRatio)(pool4)
    # convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_conv_block2(start_neurons * 16, 21, 99)(conv4)

    # Middle
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    # conv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = scse_block2(REDUCTION, 34, 98)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    # conv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = scse_block2(REDUCTION, 33, 98)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    # conv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = scse_block2(REDUCTION, 32, 98)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    # conv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = scse_block2(REDUCTION, 31, 98)(uconv1)

    # seg part
    output_seg008 = Conv2D(1, (1, 1), padding="same", name="output_seg008")(BatchNormalization(name="output_seg008_bn")(convm))
    output_seg016 = Conv2D(1, (1, 1), padding="same", name="output_seg016")(BatchNormalization(name="output_seg016_bn")(uconv4))
    output_seg032 = Conv2D(1, (1, 1), padding="same", name="output_seg032")(BatchNormalization(name="output_seg032_bn")(uconv3))
    output_seg064 = Conv2D(1, (1, 1), padding="same", name="output_seg064")(BatchNormalization(name="output_seg064_bn")(uconv2))
    output_seg128 = Conv2D(1, (1, 1), padding="same", name="output_seg128")(BatchNormalization(name="output_seg128_bn")(uconv1))

    # clf part
    output_clf = classifier_block(3, start_neurons, "output_clf")(convm)
    fusion_img = fusion_block(int(uconv1.shape[1]))(output_clf)

    # fusion part
    fusion_layer = Concatenate()([
        fusion_img,
        Multiply()([output_clf, output_seg128]),
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4),
        UpSampling2D(size=(16, 16), interpolation="bilinear")(convm)
    ])

    fusion_layer = BatchNormalization(name="output_bn0")(fusion_layer)
    fusion_layer = Conv2D(start_neurons, (3, 3), padding="same", name="output_conv1")(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn1")(fusion_layer)  # not run
    fusion_layer = Activation(ACTIVATION)(fusion_layer)
    output_fusion = Conv2D(1, (1, 1), padding="same", name="output_fusion")(fusion_layer)

    return [output_fusion,
            output_seg128,
            output_seg064,
            output_seg032,
            output_seg016,
            output_seg008,
            output_clf]


# ### 3-tasks5 deep + scse + conv

# In[80]:

ACTIVATION = "relu"
REDUCTION = 8


def _build_unet34magicbasic_multi5(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 128
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = scse_block2(REDUCTION, 11, 2)(conv1)

    # 128 -> 64
    # pool1 = MaxPooling2D((2, 2))(conv1)
    # pool1 = Dropout(DropoutRatio / 2)(pool1)
    # conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_conv_block2(start_neurons * 2, 12, 99)(conv1)

    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = scse_block2(REDUCTION, 12, 2)(conv2)

    # 64 -> 32
    # pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(DropoutRatio)(pool2)
    # conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_conv_block2(start_neurons * 4, 13, 99)(conv2)

    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = scse_block2(REDUCTION, 13, 2)(conv3)

    # 32 -> 16
    # pool3 = MaxPooling2D((2, 2))(conv3)
    # pool3 = Dropout(DropoutRatio)(pool3)
    # conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_conv_block2(start_neurons * 8, 14, 99)(conv3)

    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = scse_block2(REDUCTION, 14, 2)(conv4)

    # 16 -> 8
    # pool4 = MaxPooling2D((2, 2))(conv4)
    # pool4 = Dropout(DropoutRatio)(pool4)
    # convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_conv_block2(start_neurons * 16, 21, 99)(conv4)

    # Middle
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    # conv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = scse_block2(REDUCTION, 34, 2)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    # conv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = scse_block2(REDUCTION, 33, 2)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    # conv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = scse_block2(REDUCTION, 32, 2)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    # conv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = scse_block2(REDUCTION, 31, 2)(uconv1)

    # seg part
    output_seg008 = Conv2D(1, (1, 1), padding="same", name="output_seg008")(BatchNormalization(name="output_seg008_bn")(convm))
    output_seg016 = Conv2D(1, (1, 1), padding="same", name="output_seg016")(BatchNormalization(name="output_seg016_bn")(uconv4))
    output_seg032 = Conv2D(1, (1, 1), padding="same", name="output_seg032")(BatchNormalization(name="output_seg032_bn")(uconv3))
    output_seg064 = Conv2D(1, (1, 1), padding="same", name="output_seg064")(BatchNormalization(name="output_seg064_bn")(uconv2))
    output_seg128 = Conv2D(1, (1, 1), padding="same", name="output_seg128")(BatchNormalization(name="output_seg128_bn")(uconv1))

    # clf part
    output_clf = classifier_block(3, start_neurons, "output_clf")(convm)
    fusion_img = fusion_block(int(uconv1.shape[1]))(output_clf)

    # fusion part
    fusion_layer = Concatenate()([
        fusion_img,
        Multiply()([output_clf, output_seg128]),
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])

    fusion_layer = BatchNormalization(name="output_bn0")(fusion_layer)
    fusion_layer = Conv2D(start_neurons, (3, 3), padding="same", name="output_conv1")(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn1")(fusion_layer)  # not run
    fusion_layer = Activation(ACTIVATION)(fusion_layer)
    output_fusion = Conv2D(1, (1, 1), padding="same", name="output_fusion")(fusion_layer)

    return [output_fusion,
            output_seg128,
            output_seg064,
            output_seg032,
            output_seg016,
            output_seg008,
            output_clf]


# ### 3-task4 deep + scse

# In[63]:

ACTIVATION = "relu"
REDUCTION = 8


def _build_unet34magicbasic_multi4(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = scse_block2(REDUCTION, 11, 2)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    # pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = scse_block2(REDUCTION, 12, 2)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = scse_block2(REDUCTION, 13, 2)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    # pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = scse_block2(REDUCTION, 14, 2)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    # pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    # conv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = scse_block2(REDUCTION, 34, 2)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    # conv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = scse_block2(REDUCTION, 33, 2)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    # conv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = scse_block2(REDUCTION, 32, 2)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    # conv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = scse_block2(REDUCTION, 31, 2)(uconv1)

    # seg part
    output_seg008 = Conv2D(1, (1, 1), padding="same", name="output_seg008")(BatchNormalization(name="output_seg008_bn")(convm))
    output_seg016 = Conv2D(1, (1, 1), padding="same", name="output_seg016")(BatchNormalization(name="output_seg016_bn")(uconv4))
    output_seg032 = Conv2D(1, (1, 1), padding="same", name="output_seg032")(BatchNormalization(name="output_seg032_bn")(uconv3))
    output_seg064 = Conv2D(1, (1, 1), padding="same", name="output_seg064")(BatchNormalization(name="output_seg064_bn")(uconv2))
    output_seg128 = Conv2D(1, (1, 1), padding="same", name="output_seg128")(BatchNormalization(name="output_seg128_bn")(uconv1))

    # clf part
    output_clf = classifier_block(3, start_neurons, "output_clf")(convm)
    fusion_img = fusion_block(int(uconv1.shape[1]))(output_clf)

    # fusion part
    fusion_layer = Concatenate()([
        fusion_img,
        Multiply()([output_clf, output_seg128]),
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])

    fusion_layer = BatchNormalization(name="output_bn0")(fusion_layer)
    fusion_layer = Conv2D(start_neurons, (3, 3), padding="same", name="output_conv1")(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn1")(fusion_layer)  # not run
    fusion_layer = Activation(ACTIVATION)(fusion_layer)
    output_fusion = Conv2D(1, (1, 1), padding="same", name="output_fusion")(fusion_layer)

    return [output_fusion,
            output_seg128,
            output_seg064,
            output_seg032,
            output_seg016,
            output_seg008,
            output_clf]


# ### 3-task3-5 deep

# In[60]:

ACTIVATION = "relu"

def _build_unet34magicbasic_multi3_5(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    # pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    # pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    # pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    # conv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    # conv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    # conv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    # conv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    # seg part
    output_seg008 = Conv2D(1, (1, 1), padding="same", name="output_seg008")(BatchNormalization(name="output_seg008_bn")(convm))
    output_seg016 = Conv2D(1, (1, 1), padding="same", name="output_seg016")(BatchNormalization(name="output_seg016_bn")(uconv4))
    output_seg032 = Conv2D(1, (1, 1), padding="same", name="output_seg032")(BatchNormalization(name="output_seg032_bn")(uconv3))
    output_seg064 = Conv2D(1, (1, 1), padding="same", name="output_seg064")(BatchNormalization(name="output_seg064_bn")(uconv2))
    output_seg128 = Conv2D(1, (1, 1), padding="same", name="output_seg128")(BatchNormalization(name="output_seg128_bn")(uconv1))

    # clf part
    output_clf = classifier_block(3, start_neurons, "output_clf")(convm)
    fusion_img = fusion_block(int(uconv1.shape[1]))(output_clf)

    # fusion part
    fusion_layer = Concatenate()([
        fusion_img,
        Multiply()([output_clf, output_seg128]),
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])

    fusion_layer = BatchNormalization(name="output_bn0")(fusion_layer)
    fusion_layer = Conv2D(start_neurons, (3, 3), padding="same", name="output_conv1")(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn1")(fusion_layer)  # not run
    fusion_layer = Activation(ACTIVATION)(fusion_layer)
    output_fusion = Conv2D(1, (1, 1), padding="same", name="output_fusion")(fusion_layer)

    return [output_fusion,
            output_seg128,
            output_seg064,
            output_seg032,
            output_seg016,
            output_seg008,
            output_clf]


# ### 3-task3 deep netowrk

# In[48]:

ACTIVATION = "relu"

def _build_unet34magicbasic_multi3(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    # seg part
    output_seg008 = Conv2D(1, (1, 1), padding="same", name="output_seg008")(BatchNormalization(name="output_seg008_bn")(convm))
    output_seg016 = Conv2D(1, (1, 1), padding="same", name="output_seg016")(BatchNormalization(name="output_seg016_bn")(uconv4))
    output_seg032 = Conv2D(1, (1, 1), padding="same", name="output_seg032")(BatchNormalization(name="output_seg032_bn")(uconv3))
    output_seg064 = Conv2D(1, (1, 1), padding="same", name="output_seg064")(BatchNormalization(name="output_seg064_bn")(uconv2))
    output_seg128 = Conv2D(1, (1, 1), padding="same", name="output_seg128")(BatchNormalization(name="output_seg128_bn")(uconv1))
    
    # clf part
    output_clf = classifier_block(3, start_neurons, "output_clf")(convm)
    fusion_img = fusion_block(int(uconv1.shape[1]))(output_clf)

    # fusion part
    fusion_layer = Concatenate()([
        UpSampling2D(size=(16, 16), interpolation="bilinear")(output_seg008),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(output_seg016),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(output_seg032),
        UpSampling2D(size=(2, 2), interpolation="bilinear")(output_seg064),
        output_seg128,
        fusion_img,
        Multiply()([output_clf, output_seg128]),
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])

    fusion_layer = BatchNormalization(name="output_bn0")(fusion_layer)
    fusion_layer = Conv2D(start_neurons, (3, 3), padding="same", name="output_conv1")(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn1")(fusion_layer)  # not run
    fusion_layer = Activation(ACTIVATION)(fusion_layer)
    output_fusion = Conv2D(1, (1, 1), padding="same", name="output_fusion")(fusion_layer)

    return [output_fusion,
            output_seg128,
            output_seg064,
            output_seg032,
            output_seg016,
            output_seg008,
            output_clf]


# ### 3-task2-5 network

# In[44]:

ACTIVATION = "relu"

def _build_unet34magicbasic_multi2_5(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    # seg layer
    seg_layer = Concatenate()([
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4),
    ])

    seg_layer = BatchNormalization(name="output_seg_bn0")(seg_layer)
    seg_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_seg_conv1")(seg_layer)
    seg_layer = BatchNormalization(name="output_seg_bn1")(seg_layer)  # not run
    seg_layer = Activation(ACTIVATION)(seg_layer)
    seg_output = Conv2D(1, (1, 1), padding="same", name="output_seg")(seg_layer)

    clf_output = classifier_block(3, start_neurons, "output_clf")(convm)
    fusion_img = fusion_block(int(uconv1.shape[1]))(clf_output)

    # fusion layer
    fusion_layer = Concatenate()([
        fusion_img,
        Multiply()([clf_output, seg_output]),
        seg_output
    ])

    fusion_layer = BatchNormalization(name="output_bn0")(fusion_layer)
    fusion_layer = Conv2D(start_neurons * 2, (3, 3), padding="same", name="output_conv1")(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn1")(fusion_layer)  # not run
    fusion_layer = Activation(ACTIVATION)(fusion_layer)
    fusion_output = Conv2D(1, (1, 1), padding="same", name="output_fusion")(fusion_layer)

    if use_sigmoid:
        seg_output = Activation("sigmoid", name="output_seg_sigmoid")(seg_output)
        fusion_output = Activation("sigmoid", name="output_fusion_sigmoid")(fusion_output)

    return [seg_output, clf_output, fusion_output]


# ### 3-task2 network

# In[24]:

ACTIVATION = "relu"

def _build_unet34magicbasic_multi2(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    # seg layer
    seg_layer = Concatenate()([
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4),
    ])

    seg_layer = BatchNormalization(name="output_seg_bn0")(seg_layer)
    seg_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_seg_conv1")(seg_layer)
    seg_layer = BatchNormalization(name="output_seg_bn1")(seg_layer)  # not run
    seg_layer = Activation(ACTIVATION)(seg_layer)
    seg_output = Conv2D(1, (1, 1), padding="same", name="output_seg")(seg_layer)

    clf_output = classifier_block(3, start_neurons, "output_clf")(convm)
    fusion_img = fusion_block(int(uconv1.shape[1]))(clf_output)

    # fusion layer
    fusion_layer = Concatenate()([
        uconv1,       # remove?
        fusion_img,
        # Multiply()([clf_output, seg_output]),
        seg_output
    ])

    fusion_layer = BatchNormalization(name="output_bn0")(fusion_layer)
    fusion_layer = Conv2D(start_neurons * 2, (3, 3), padding="same", name="output_conv1")(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn1")(fusion_layer)  # not run
    fusion_layer = Activation(ACTIVATION)(fusion_layer)
    fusion_output = Conv2D(1, (1, 1), padding="same", name="output_fusion")(fusion_layer)

    if use_sigmoid:
        seg_output = Activation("sigmoid", name="output_seg_sigmoid")(seg_output)
        fusion_output = Activation("sigmoid", name="output_fusion_sigmoid")(fusion_output)

    return [seg_output, clf_output, fusion_output]


# ### 3-task network

# In[15]:

ACTIVATION = "relu"

def _build_unet34magicbasic_multi(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    seg_output = Conv2D(1, (1, 1), padding="same", name="output_seg")(uconv1)
    clf_output = classifier_block(3, start_neurons, "output_clf")(convm)
    fusion_img = fusion_block(int(uconv1.shape[1]))(clf_output)

    # fusion layer
    fusion_layer = Concatenate()([
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4),
        fusion_img
    ])

    # fusion_layer = Dropout(DropoutRatio / 2)(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn0")(fusion_layer)
    # fusion_layer = Activation(ACTIVATION)(fusion_layer)
    fusion_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(fusion_layer)
    fusion_layer = BatchNormalization(name="output_bn1")(fusion_layer)  # not run
    # fusion_layer = Dropout(DropoutRatio / 2)(fusion_layer)
    fusion_layer = Activation(ACTIVATION)(fusion_layer)
    fusion_output = Conv2D(1, (1, 1), padding="same", name="output_fusion")(fusion_layer)

    if use_sigmoid:
        seg_output = Activation("sigmoid", name="output_seg_sigmoid")(seg_output)
        fusion_output = Activation("sigmoid", name="output_fusion_sigmoid")(fusion_output)

    return [seg_output, clf_output, fusion_output]


# ### unet256 + resnext

# In[14]:

ACTIVATION = "relu"


def _build_unet256(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 256 -> 128
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = scse_block(start_neurons * 1, 11, 1)(conv1)

    # 128 -> 64
    conv2 = conv_block(start_neurons * 1, 11, 2)(conv1)
    conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = scse_block(start_neurons * 2, 12, 1)(conv2)

    # 64 -> 32
    conv3 = conv_block(start_neurons * 2, 12, 2)(conv2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = scse_block(start_neurons * 4, 13, 1)(conv3)

    # 32 -> 16
    conv4 = conv_block(start_neurons * 4, 13, 2)(conv3)
    conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = scse_block(start_neurons * 8, 14, 1)(conv4)

    # Middle
    convm = conv_block(start_neurons * 8, 14, 2)(conv4)
    convm = identity_block(start_neurons * 16, 21, 0)(convm)
    convm = scse_block(start_neurons * 16, 21, 1)(convm)

    # 16 -> 32
    deconv4 = UpSampling2D()(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Activation(ACTIVATION)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = scse_block(start_neurons * 8, 34, 1)(uconv4)
    # uconv4 = scse_block(start_neurons * 8, 34, 2)(uconv4)

    # 32 -> 64
    deconv3 = UpSampling2D()(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Activation(ACTIVATION)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = scse_block(start_neurons * 4, 33, 1)(uconv3)
    # uconv3 = scse_block(start_neurons * 4, 33, 2)(uconv3)

    # 64 -> 128
    deconv2 = UpSampling2D()(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Activation(ACTIVATION)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = scse_block(start_neurons * 2, 32, 1)(uconv2)
    # uconv2 = scse_block(start_neurons * 2, 32, 2)(uconv2)

    # 128 -> 256
    deconv1 = UpSampling2D()(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Activation(ACTIVATION)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = scse_block(start_neurons * 1, 31, 1)(uconv1)
    # uconv1 = scse_block(start_neurons * 1, 31, 2)(uconv1)

    # magic layer
    output_layer = Concatenate()([
        uconv1,
        UpSampling2D()(uconv2),
        UpSampling2D(size=(4, 4))(uconv3),
        UpSampling2D(size=(8, 8))(uconv4)
    ])
    
    # output layer
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    output_layer = BatchNormalization(name="output_bn")(output_layer) # not run
    # output_layer = Dropout(DropoutRatio // 2)(output_layer)    # not run
    output_layer = Dropout(DropoutRatio)(output_layer)
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", name="output_conv2")(output_layer)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet256ref + resnext

# In[15]:

ACTIVATION = "relu"


def _build_unet256_ref(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 256 -> 128
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    # conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    # pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 128 -> 64
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    # conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(DropoutRatio / 2)(pool2)

    # 64 -> 32
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    # conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    # pool3 = Dropout(DropoutRatio / 2)(pool3)

    # 32 -> 16
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    # conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio / 2)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = identity_block(start_neurons * 16, 21, 0)(convm)
    convm = identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 16 -> 32
    deconv4 = UpSampling2D()(convm)
    uconv4 = concatenate([deconv4, conv4])

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 0)(uconv4)
    # uconv4 = identity_block(start_neurons * 8, 34, 1)(uconv4)
    # uconv4 = scse_block(start_neurons * 8, 34, 2)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 32 -> 64
    deconv3 = UpSampling2D()(uconv4)
    uconv3 = concatenate([deconv3, conv3])

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    # uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    # uconv3 = scse_block(start_neurons * 4, 33, 2)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 64 -> 128
    deconv2 = UpSampling2D()(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    # uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    # uconv2 = scse_block(start_neurons * 2, 32, 2)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)
    
    # 128 -> 256
    deconv1 = UpSampling2D()(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    # uconv1 = identity_block(start_neurons * 1, 31, 1)(uconv1)
    # uconv1 = scse_block(start_neurons * 1, 31, 2)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)
    
    # magic layer
    '''
    output_layer = Concatenate()([
        uconv1,
        UpSampling2D()(uconv2),
        UpSampling2D(size=(4, 4))(uconv3),
        UpSampling2D(size=(8, 8))(uconv4)
    ])
    '''

    # output layer

    # output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    # output_layer = BatchNormalization(name="output_bn")(output_layer) # not run
    # output_layer = Dropout(DropoutRatio // 2)(output_layer)    # not run
    output_layer = Dropout(DropoutRatio)(uconv1)
    # output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (3, 3), padding="same", name="output_conv2")(output_layer)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet34 + resnet

# In[18]:

ACTIVATION = "relu"


def _build_unet34(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
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
    conv2 = identity_block(start_neurons * 2, 12, 3)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 2)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 3)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 4)(conv3)
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
    uconv3 = identity_block(start_neurons * 4, 33, 3)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 4)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 2)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 3)(uconv2)
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


# ### unet34 for 128 & 256 + resnet

# In[17]:

ACTIVATION = "relu"


def _build_unet34_for256(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
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
    conv2 = identity_block(start_neurons * 2, 12, 3)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 2)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 3)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 4)(conv3)
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
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 2)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 3)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 4)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 2)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 3)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
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


# ### unet34magic for 128 & 256 + resnet

# In[18]:

def _build_unet34magic_for256(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = identity_block(start_neurons * 8, 14, 1)(conv4)
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
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    output_layer = Concatenate()([
        uconv1,
        UpSampling2D()(uconv2),
        UpSampling2D(size=(4, 4))(uconv3),
        UpSampling2D(size=(8, 8))(uconv4)
    ])
    
    # output layer
    output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = BatchNormalization(name="output_bn0")(output_layer)
    # output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    output_layer = BatchNormalization(name="output_bn1")(output_layer) # not run
    # output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet34magicfixed for 128

# In[19]:

def _build_unet34magicfixed_for256(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = identity_block(start_neurons * 8, 14, 1)(conv4)
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
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    output_layer = Concatenate()([
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])
    
    # output layer
    output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = BatchNormalization(name="output_bn0")(output_layer)
    # output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    output_layer = BatchNormalization(name="output_bn1")(output_layer) # not run
    # output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet34basicmagic for 128

# In[20]:

def _build_unet34magicbasic_for256(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    output_layer = Concatenate()([
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])
    
    # output layer
    output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = BatchNormalization(name="output_bn0")(output_layer)
    # output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    output_layer = BatchNormalization(name="output_bn1")(output_layer) # not run
    # output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet34basicmagic for 128 mix drop

# In[ ]:

def _build_unet34magicbasic_for256_mixdrop(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = SpatialDropout2D(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = SpatialDropout2D(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = SpatialDropout2D(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = SpatialDropout2D(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    output_layer = Concatenate()([
        SpatialDropout2D(DropoutRatio / 2)(uconv1),
        SpatialDropout2D(DropoutRatio)(UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2)),
        SpatialDropout2D(DropoutRatio)(UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3)),
        SpatialDropout2D(DropoutRatio)(UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4))
    ])
    
    # output layer
    # output_layer = SpatialDropout2D(DropoutRatio / 2)(output_layer)
    output_layer = BatchNormalization(name="output_bn0")(output_layer)
    # output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    output_layer = BatchNormalization(name="output_bn1")(output_layer) # not run
    # output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet34basicscse-magic for 128

# In[21]:

REDUCTION = 4

def _build_unet34magicscse_for256(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = scse_block(REDUCTION, 11, 2)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = scse_block(REDUCTION * 2, 12, 2)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = scse_block(REDUCTION * 4, 13, 2)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = scse_block(REDUCTION * 8, 14, 2)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = scse_block(REDUCTION * 8, 34, 2)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = scse_block(REDUCTION * 4, 33, 2)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = scse_block(REDUCTION * 2, 32, 2)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = scse_block(REDUCTION, 31, 2)(uconv1)

    output_layer = Concatenate()([
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])
    
    # output layer
    output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = BatchNormalization(name="output_bn0")(output_layer)
    # output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    output_layer = BatchNormalization(name="output_bn1")(output_layer) # not run
    # output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet34basicscse-magic for 128 mix drop

# In[90]:

REDUCTION = 4

def _build_unet34magicscse_for256_mixdrop(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    # conv1 = scse_block(REDUCTION, 11, 2)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    # conv2 = scse_block(REDUCTION * 2, 12, 2)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    # conv3 = scse_block(REDUCTION * 4, 13, 2)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Activation(ACTIVATION)(conv4)
    # conv4 = scse_block(REDUCTION * 8, 14, 2)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = scse_block(REDUCTION * 8, 34, 2)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = scse_block(REDUCTION * 4, 33, 2)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = scse_block(REDUCTION * 2, 32, 2)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = scse_block(REDUCTION, 31, 2)(uconv1)

    output_layer = Concatenate()([
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])
    
    # output layer
    # output_layer = SpatialDropout2D(DropoutRatio)(output_layer)
    output_layer = BatchNormalization(name="output_bn0")(output_layer)
    # output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    output_layer = BatchNormalization(name="output_bn1")(output_layer) # not run
    # output_layer = SpatialDropout2D(DropoutRatio / 2)(output_layer)
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet34basicscse-magic-dropmore for 128 & 256 + resnet

# In[23]:

REDUCTION = 4

def _build_unet34magicscse_for256_dropmore(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = basic_identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = basic_identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Dropout(DropoutRatio / 2)(conv1)
    conv1 = scse_block(REDUCTION, 11, 2)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = basic_identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = basic_identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Dropout(DropoutRatio / 2)(conv2)
    conv2 = scse_block(REDUCTION * 2, 12, 2)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = basic_identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = basic_identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Dropout(DropoutRatio / 2)(conv3)
    conv3 = scse_block(REDUCTION * 4, 13, 2)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = basic_identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = basic_identity_block(start_neurons * 8, 14, 1)(conv4)
    conv4 = Dropout(DropoutRatio / 2)(conv4)
    conv4 = scse_block(REDUCTION * 8, 14, 2)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same", name="vm_a_conv")(pool4)
    convm = basic_identity_block(start_neurons * 16, 21, 0)(convm)
    convm = basic_identity_block(start_neurons * 16, 21, 1)(convm)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = basic_identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Dropout(DropoutRatio / 2)(uconv4)
    uconv4 = scse_block(REDUCTION * 8, 34, 2)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = basic_identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Dropout(DropoutRatio / 2)(uconv3)
    uconv3 = scse_block(REDUCTION * 4, 33, 2)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = basic_identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Dropout(DropoutRatio / 2)(uconv2)
    uconv2 = scse_block(REDUCTION * 2, 32, 2)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = basic_identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    uconv1 = scse_block(REDUCTION, 31, 2)(uconv1)

    output_layer = Concatenate()([
        uconv1,
        UpSampling2D(size=(2, 2), interpolation="bilinear")(uconv2),
        UpSampling2D(size=(4, 4), interpolation="bilinear")(uconv3),
        UpSampling2D(size=(8, 8), interpolation="bilinear")(uconv4)
    ])
    
    # output layer
    output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = BatchNormalization(name="output_bn0")(output_layer)
    # output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(start_neurons * 4, (3, 3), padding="same", name="output_conv1")(output_layer)
    output_layer = BatchNormalization(name="output_bn1")(output_layer) # not run
    # output_layer = Dropout(DropoutRatio / 2)(output_layer)
    output_layer = Activation(ACTIVATION)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### thinunet + resnext

# In[24]:

ACTIVATION = "relu"


def _build_thinunet(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = identity_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = identity_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = identity_block(start_neurons * 8, 14, 1)(conv4)
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
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### link-net + resnext

# In[25]:

def _build_linknet(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
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
    uconv4 = Add()([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="de4_a_conv")(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 0)(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = identity_block(start_neurons * 8, 34, 2)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = Add()([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 2)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = Add()([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 2)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = Add()([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 2)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    # uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### unet + resnext + scse block

# In[26]:

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


# ### thinunent + resnext + scse-block

# In[27]:

def _build_unet_scse(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="en1_a_conv")(input_layer)
    conv1 = identity_block(start_neurons * 1, 11, 0)(conv1)
    conv1 = scse_block(start_neurons * 1, 11, 1)(conv1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="en2_a_conv")(pool1)
    conv2 = identity_block(start_neurons * 2, 12, 0)(conv2)
    conv2 = scse_block(start_neurons * 2, 12, 1)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = scse_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same", name="en4_a_conv")(pool3)
    conv4 = identity_block(start_neurons * 8, 14, 0)(conv4)
    conv4 = scse_block(start_neurons * 8, 14, 1)(conv4)
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
    uconv4 = scse_block(start_neurons * 8, 34, 1)(uconv4)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="de3_a_conv")(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 0)(uconv3)
    uconv3 = scse_block(start_neurons * 4, 33, 1)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = scse_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", name="de1_a_conv")(uconv1)
    uconv1 = identity_block(start_neurons * 1, 31, 0)(uconv1)
    uconv1 = scse_block(start_neurons * 1, 31, 1)(uconv1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", name="final_conv")(uconv1)

    if use_sigmoid:
        output_layer = Activation("sigmoid")(output_layer)

    return output_layer


# ### link-net + resnext + scse-block

# In[28]:

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


# ### unet34 + resnext + scse-block

# In[29]:

ACTIVATION = "relu"


def _build_unet34_scse(input_layer, start_neurons, DropoutRatio=0.5, use_sigmoid=False):
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
    conv2 = identity_block(start_neurons * 2, 12, 2)(conv2)
    conv2 = scse_block(start_neurons * 2, 12, 3)(conv2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", name="en3_a_conv")(pool2)
    conv3 = identity_block(start_neurons * 4, 13, 0)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 1)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 2)(conv3)
    conv3 = identity_block(start_neurons * 4, 13, 3)(conv3)
    conv3 = scse_block(start_neurons * 4, 13, 4)(conv3)
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
    uconv3 = identity_block(start_neurons * 4, 33, 2)(uconv3)
    uconv3 = identity_block(start_neurons * 4, 33, 3)(uconv3)
    uconv3 = scse_block(start_neurons * 4, 33, 4)(uconv3)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", name="de2_a_conv")(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 0)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 1)(uconv2)
    uconv2 = identity_block(start_neurons * 2, 32, 2)(uconv2)
    uconv2 = scse_block(start_neurons * 2, 32, 3)(uconv2)
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


# ## model builders

# In[25]:

def build_stage0_model_multi(img_size_target,
                             builder=_build_unet34magicbasic_multi2,
                             loss={'output_fusion': bce_lovasz_loss,
                                   'output_seg': bce_lovasz_loss_nonzero,
                                   'output_clf': "categorical_crossentropy"},
                             loss_weights={'output_fusion': 1.,
                                           'output_seg': .1,
                                           'output_clf': 0.05},
                             metrics={'output_fusion': my_iou_metric,
                                      'output_seg': my_iou_metric,
                                      'output_clf': ["accuracy"]},
                             opt="adam",
                             use_sigmoid=False):
    print("local: model builders multi-task")
    input_layer = Input((img_size_target, img_size_target, 2), name="input")
    output_layers = builder(input_layer, 16, 0.5, use_sigmoid=use_sigmoid)
    model = Model(input_layer, output_layers)

    model.compile(loss=loss,
                  loss_weights=loss_weights,
                  optimizer=opt,
                  metrics=metrics)
    return model


# In[19]:

def build_model(img_size_target, builder=_build_unet34):
    input_layer = Input((img_size_target, img_size_target, 2), name="input")
    output_layer = builder(input_layer, 16, 0.5, use_sigmoid=True)

    model = Model(input_layer, output_layer)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[my_iou_metric_2])
    return model


# In[20]:

def build_stage0_model(img_size_target, builder=_build_unet34, loss=focal_loss, metric=my_iou_metric_2, opt="adam", use_sigmoid=False):
    print("local: model builders")
    input_layer = Input((img_size_target, img_size_target, 2), name="input")
    output_layer = builder(input_layer, 16, 0.5, use_sigmoid=use_sigmoid)
    model = Model(input_layer, output_layer)

    model.compile(loss=loss, optimizer=opt, metrics=[metric])
    return model


# In[21]:

def get_model_params(loss_name):
    stack = {
        "bce": {"loss": my_bce, "use_sigmoid": False, "metric": my_iou_metric},
        "bcedice": {"loss": bce_dice_loss, "use_sigmoid": False, "metric": my_iou_metric},
        "bcelovasz": {"loss": bce_lovasz_loss, "use_sigmoid": False, "metric": my_iou_metric},
        "focal": {"loss": focal_loss, "use_sigmoid": True, "metric": my_iou_metric_2},
        "lovasz": {"loss": lovasz_loss, "use_sigmoid": False, "metric": my_iou_metric}
    }
    
    metric_names = {
        "bce": "my_iou_metric",
        "bcedice": "my_iou_metric",
        "bcelovasz": "my_iou_metric",
        "focal": "my_iou_metric_2",
        "lovasz": "my_iou_metric"
    }

    return stack.get(loss_name), metric_names.get(loss_name)


# In[50]:

def build_stage0_model_deep(img_size_target,
                             builder=_build_unet34magicbasic_multi3,
                             loss={'output_fusion': bce_lovasz_loss,
                                   'output_seg128': bce_lovasz_loss_nonzero,
                                   'output_seg064': bce_lovasz_loss_nonzero,
                                   'output_seg032': bce_lovasz_loss_nonzero,
                                   'output_seg016': bce_lovasz_loss_nonzero,
                                   'output_seg008': bce_lovasz_loss_nonzero,
                                   'output_clf': "categorical_crossentropy"},
                             loss_weights={'output_fusion': 1.,
                                           'output_seg128': .02,
                                           'output_seg064': .02,
                                           'output_seg032': .02,
                                           'output_seg016': .02,
                                           'output_seg008': .02,
                                           'output_clf': .05},
                             metrics={'output_fusion': my_iou_metric,
                                      'output_seg128': my_iou_metric,
                                      'output_seg064': my_iou_metric,
                                      'output_seg032': my_iou_metric,
                                      'output_seg016': my_iou_metric,
                                      'output_seg008': my_iou_metric,                                      
                                      'output_clf': "accuracy"},
                             opt="adam",
                             use_sigmoid=False):
    print("local: model builders multi-task deep")
    input_layer = Input((img_size_target, img_size_target, 2), name="input")
    output_layers = builder(input_layer, 16, 0.5, use_sigmoid=use_sigmoid)
    model = Model(input_layer, output_layers)

    model.compile(loss=loss,
                  loss_weights=loss_weights,
                  optimizer=opt,
                  metrics=metrics)
    return model


def load_model_bce2lovasz(filename):
    co = {
        "my_iou_metric_2": my_iou_metric_2
    }
    model_pretrained = load_model(filename, custom_objects=co)
    model_rebuild = Model(model_pretrained.layers[0].input, model_pretrained.layers[-1].input)

    c = SGD(lr=0.001, momentum=.9, decay=1e-6)
    model_rebuild.compile(loss=lovasz_loss_elu, optimizer=c, metrics=[my_iou_metric])
    return model_rebuild


# In[46]:

def load_model_mybce2lovasz(filename, opt):
    co = {
        "my_bce": my_bce,
        "my_iou_metric": my_iou_metric
    }
    model_pretrained = load_model(filename, custom_objects=co)
    model_pretrained.compile(loss=lovasz_loss_elu, optimizer=opt, metrics=[my_iou_metric])
    return model_pretrained


# In[63]:

def load_model_bcelovasz2lovasz(filename, opt):
    co = {
        "bce_lovasz_loss": bce_lovasz_loss,
        "my_iou_metric": my_iou_metric
    }
    model_pretrained = load_model(filename, custom_objects=co)
    model_pretrained.compile(loss=lovasz_loss_elu, optimizer=opt, metrics=[my_iou_metric])
    return model_pretrained


# In[47]:

def load_model_dice2lovasz(filename, opt):
    co = {
        "bce_dice_loss": bce_dice_loss,
        "my_iou_metric": my_iou_metric,
        "my_iou_metric_2": my_iou_metric_2
    }
    
    model_pretrained = load_model(filename, custom_objects=co)
    print(model_pretrained.layers[-1].get_config())
    model_pretrained.compile(loss=lovasz_loss_elu, optimizer=opt, metrics=[my_iou_metric])
    return model_pretrained


# In[112]:

def load_model_bcelovasz2lovasz_multi(filename, opt):
    co = {
        "bce_lovasz_loss": bce_lovasz_loss,
        "bce_lovasz_loss_nonzero": bce_lovasz_loss_nonzero,
        "my_iou_metric": my_iou_metric
    }
    model_pretrained = load_model(filename, custom_objects=co)
    model_pretrained.compile(loss={'output_fusion': lovasz_loss_elu,
                                   'output_seg': lovasz_loss_elu_nonzero,
                                   'output_clf': "categorical_crossentropy"},
                             loss_weights={'output_fusion': 1.,
                                           'output_seg': .1,
                                           'output_clf': 0.05},
                             metrics={'output_fusion': my_iou_metric,
                                      'output_seg': my_iou_metric,
                                      'output_clf': ["accuracy"]},
                             optimizer=opt)
    return model_pretrained


# In[88]:

def load_model_bcelovasz2lovasz_deep(filename, opt):
    co = {
        "bce_lovasz_loss": bce_lovasz_loss,
        "bce_lovasz_loss_nonzero": bce_lovasz_loss_nonzero,
        "lovasz_loss_elu": lovasz_loss_elu,
        "lovasz_loss_elu_nonzero": lovasz_loss_elu_nonzero,
        "my_iou_metric": my_iou_metric
    }

    model_params = {
        "loss": {
            'output_fusion': lovasz_loss_elu,
            'output_seg128': lovasz_loss_elu_nonzero,
            'output_seg064': lovasz_loss_elu_nonzero,
            'output_seg032': lovasz_loss_elu_nonzero,
            'output_seg016': lovasz_loss_elu_nonzero,
            'output_seg008': lovasz_loss_elu_nonzero,
            'output_clf': "categorical_crossentropy"
        },
        "loss_weights": {
            'output_fusion': 1.,
            'output_seg128': .02,
            'output_seg064': .02,
            'output_seg032': .02,
            'output_seg016': .02,
            'output_seg008': .02,
            'output_clf': .05
        },
        "metrics": {
            'output_fusion': my_iou_metric,
            'output_seg128': my_iou_metric,
            'output_seg064': my_iou_metric,
            'output_seg032': my_iou_metric,
            'output_seg016': my_iou_metric,
            'output_seg008': my_iou_metric,
            'output_clf': "accuracy"
        },
        "optimizer": opt
    }
    model_pretrained = load_model(filename, custom_objects=co)
    model_pretrained.compile(**model_params)
    return model_pretrained


def load_model_with_allco(filename):
    co = {
        "bce_lovasz_loss": bce_lovasz_loss,
        "bce_lovasz_loss_nonzero": bce_lovasz_loss_nonzero,
        "lovasz_loss_elu": lovasz_loss_elu,
        "lovasz_loss_elu_nonzero": lovasz_loss_elu_nonzero,
        "my_iou_metric": my_iou_metric
    }
    model_pretrained = load_model(filename, custom_objects=co)
    return model_pretrained


def change_deep_opt(model, opt):
    model_params = {
        "loss": {
            'output_fusion': lovasz_loss_elu,
            'output_seg128': lovasz_loss_elu_nonzero,
            'output_seg064': lovasz_loss_elu_nonzero,
            'output_seg032': lovasz_loss_elu_nonzero,
            'output_seg016': lovasz_loss_elu_nonzero,
            'output_seg008': lovasz_loss_elu_nonzero,
            'output_clf': "categorical_crossentropy"
        },
        "loss_weights": {
            'output_fusion': 1.,
            'output_seg128': .02,
            'output_seg064': .02,
            'output_seg032': .02,
            'output_seg016': .02,
            'output_seg008': .02,
            'output_clf': .05
        },
        "metrics": {
            'output_fusion': my_iou_metric,
            'output_seg128': my_iou_metric,
            'output_seg064': my_iou_metric,
            'output_seg032': my_iou_metric,
            'output_seg016': my_iou_metric,
            'output_seg008': my_iou_metric,
            'output_clf': "accuracy"
        },
        "optimizer": opt
    }
    model.compile(**model_params)
    return model


def change_loss2mse(model, opt):
    model_params = {
        "loss": {
            'output_fusion': 'mean_squared_error',
            'output_seg128': 'mean_squared_error',
            'output_seg064': 'mean_squared_error',
            'output_seg032': 'mean_squared_error',
            'output_seg016': 'mean_squared_error',
            'output_seg008': 'mean_squared_error',
            'output_clf': "categorical_crossentropy"
        },
        "loss_weights": {
            'output_fusion': 1.,
            'output_seg128': .01,  # .05,  # .02,
            'output_seg064': .01,  # .05,  # .02,
            'output_seg032': .01,  # .05,  # .02,
            'output_seg016': .01,  # .05,  # .02,
            'output_seg008': .01,  # .05,  # .02,
            'output_clf': .025
        },
        "metrics": {
            'output_fusion': my_iou_metric,
            'output_seg128': my_iou_metric,
            'output_seg064': my_iou_metric,
            'output_seg032': my_iou_metric,
            'output_seg016': my_iou_metric,
            'output_seg008': my_iou_metric,
            'output_clf': "accuracy"
        },
        "optimizer": opt
    }
    model.compile(**model_params)
    return model


def change_deep_opt2(model, opt):
    model_params = {
        "loss": {
            'output_fusion': lovasz_loss_elu,
            'output_seg128': lovasz_loss_elu_nonzero,
            'output_seg064': lovasz_loss_elu_nonzero,
            'output_seg032': lovasz_loss_elu_nonzero,
            'output_seg016': lovasz_loss_elu_nonzero,
            'output_seg008': lovasz_loss_elu_nonzero,
            'output_clf': "categorical_crossentropy"
        },
        "loss_weights": {
            'output_fusion': 1.0,
            'output_seg128': .01,
            'output_seg064': .01,
            'output_seg032': .01,
            'output_seg016': .01,
            'output_seg008': .01,
            'output_clf': .025
        },
        "metrics": {
            'output_fusion': my_iou_metric,
            'output_seg128': my_iou_metric,
            'output_seg064': my_iou_metric,
            'output_seg032': my_iou_metric,
            'output_seg016': my_iou_metric,
            'output_seg008': my_iou_metric,
            'output_clf': "accuracy"
        },
        "optimizer": opt
    }
    model.compile(**model_params)
    return model


def change_deep_opt3(model, opt):
    model_params = {
        "loss": {
            'output_fusion': lovasz_loss_elu,
            'output_seg128': lovasz_loss_elu_nonzero,
            'output_seg064': lovasz_loss_elu_nonzero,
            'output_seg032': lovasz_loss_elu_nonzero,
            'output_seg016': lovasz_loss_elu_nonzero,
            'output_seg008': lovasz_loss_elu_nonzero,
            'output_clf': "categorical_crossentropy"
        },
        "loss_weights": {
            'output_fusion': 1.0,
            'output_seg128': .01,
            'output_seg064': .01,
            'output_seg032': .01,
            'output_seg016': .01,
            'output_seg008': .01,
            'output_clf': .08
        },
        "metrics": {
            'output_fusion': my_iou_metric,
            'output_seg128': my_iou_metric,
            'output_seg064': my_iou_metric,
            'output_seg032': my_iou_metric,
            'output_seg016': my_iou_metric,
            'output_seg008': my_iou_metric,
            'output_clf': "accuracy"
        },
        "optimizer": opt
    }
    model.compile(**model_params)
    return model


def change_deep_opt4(model, opt):
    model_params = {
        "loss": {
            'output_fusion': lovasz_loss_elu,
            'output_seg128': lovasz_loss_elu_nonzero,
            'output_seg064': lovasz_loss_elu_nonzero,
            'output_seg032': lovasz_loss_elu_nonzero,
            'output_seg016': lovasz_loss_elu_nonzero,
            'output_seg008': lovasz_loss_elu_nonzero,
            'output_clf': "categorical_crossentropy"
        },
        "loss_weights": {
            'output_fusion': 1.0,
            'output_seg128': .1,
            'output_seg064': .1,
            'output_seg032': .1,
            'output_seg016': .1,
            'output_seg008': .1,
            'output_clf': .05
        },
        "metrics": {
            'output_fusion': my_iou_metric,
            'output_seg128': my_iou_metric,
            'output_seg064': my_iou_metric,
            'output_seg032': my_iou_metric,
            'output_seg016': my_iou_metric,
            'output_seg008': my_iou_metric,
            'output_clf': "accuracy"
        },
        "optimizer": opt
    }
    model.compile(**model_params)
    return model
