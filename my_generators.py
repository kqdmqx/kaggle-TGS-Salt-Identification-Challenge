from my_dependency import *

# ## data augmentation & generator

# copy data augmentation code from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61949#385778

# ### get_mask_type

# In[46]:

# copy code from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984


def get_mask_type(mask, is_vertical=True):
    border = 10
    outer = np.zeros((101 - 2 * border, 101 - 2 * border), np.float32)
    outer = cv2.copyMakeBorder(
        outer, border, border, border, border, borderType=cv2.BORDER_CONSTANT, value=1)

    cover = (mask > 0.5).sum()
    if cover < 8:
        return 'empty'
    if cover == ((mask * outer) > 0.5).sum():
        return 'border'
    if np.all(mask == mask[0]) & is_vertical:
        return "vertical"

    percentage = cover / (101 * 101)

    if percentage < .15:
        return "object015"
    elif percentage < .25:
        return "object025"
    elif percentage < .5:
        return "object050"
    elif percentage < .75:
        return "object075"
    else:
        return "object100"


# ### div_round

# In[47]:

def div_round(a, b):
    result = a // b
    if b * result < a:
        result += 1
    return result


# In[48]:


# ### flip

# In[49]:

def append_flip(X):
    return np.append(X, [np.fliplr(x) for x in X], axis=0)


def do_flip(X):
    return np.array([np.fliplr(x) for x in X])


# ### reflect128

# In[50]:

def reflect128_img(x):
    return np.pad(x, ((13, 14), (13, 14), (0, 0)), mode='reflect')


def reflect128(X):
    return np.array([reflect128_img(x) for x in X])


def get_101from128(img_arr):
    return img_arr[:, 13:-14, 13:-14, :]


# ### reshape

# In[51]:

reshape256_seq = iaa.Sequential([
    iaa.Scale({'height': 202, 'width': 202}),
    PadFixed(pad=(27, 27), pad_method='reflect')
])

reshape128_seq = iaa.Scale({'height': 128, 'width': 128})


# ### augmentation

# In[52]:

affine_seq = iaa.Sequential([
    #     General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-10, 10),
                           translate_percent={"x": (-0.05, 0.05)},
                           mode='edge'),
                ]),
    #     Deformations
    iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
], random_order=True)

intensity_seq = iaa.Sequential([
    iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.Fliplr(0.5),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.Add((-10, 10)),
                iaa.AddElementwise((-10, 10)),
                iaa.Multiply((0.95, 1.05)),
                iaa.MultiplyElementwise((0.95, 1.05)),
            ]),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AverageBlur(k=(2, 5)),
            iaa.MedianBlur(k=(3, 5))
        ])
    ])
], random_order=False)

mixed_seq = iaa.Sequential([
    iaa.OneOf([
        affine_seq,
        intensity_seq
    ])
], random_order=False)


intensity_flip_seq = iaa.Sequential([
    iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.Fliplr(1.0),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.Add((-10, 10)),
                iaa.AddElementwise((-10, 10)),
                iaa.Multiply((0.95, 1.05)),
                iaa.MultiplyElementwise((0.95, 1.05)),
            ]),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AverageBlur(k=(2, 5)),
            iaa.MedianBlur(k=(3, 5))
        ])
    ])
], random_order=False)


# ### my_seq augmentation

# In[53]:

my_seq = iaa.Sequential([
    iaa.Sometimes(0.3, iaa.Affine(                                              # .3
        rotate=(-10, 10),
        translate_percent={"x": (-0.05, 0.05)},
        mode='edge'
    )
    ),
    iaa.Sometimes(0.5, iaa.PiecewiseAffine(
        scale=(0.04, 0.08))),                # .3
    iaa.Sometimes(0.5, iaa.PerspectiveTransform(
        scale=(0.05, 0.1))),            # .3
    iaa.Sometimes(0.25,                                                         # .1
                  iaa.OneOf([
                      iaa.GaussianBlur(sigma=(0.0, 1.0)),
                      iaa.AverageBlur(k=(1, 3)),
                      iaa.MedianBlur(k=(1, 3))
                  ]))
], random_order=True)


def get_my_seq(prob_list):
    p1, p2, p3, p4 = prob_list
    return iaa.Sequential([
        iaa.Sometimes(p1, iaa.Affine(                                              # .3
            rotate=(-10, 10),
            translate_percent={"x": (-0.05, 0.05)},
            mode='edge'
        )
        ),
        iaa.Sometimes(p2, iaa.PiecewiseAffine(
            scale=(0.04, 0.08))),                # .3
        iaa.Sometimes(p3, iaa.PerspectiveTransform(
            scale=(0.05, 0.1))),            # .3
        iaa.Sometimes(p4,                                                         # .1
                      iaa.OneOf([
                          iaa.GaussianBlur(sigma=(0.0, 1.0)),
                          iaa.AverageBlur(k=(1, 3)),
                          iaa.MedianBlur(k=(1, 3))
                      ]))
    ], random_order=True)


def my_augmentation(seq_det, X_train, y_train):
    X_train_aug = [(x[:, :, :] * 255.0).astype(np.uint8) for x in X_train]
    X_train_aug = seq_det.augment_images(X_train_aug)
    X_train_aug = [(x[:, :, :].astype(np.float64) / 255.0)
                   for x in X_train_aug]

    y_train_aug = [(x[:, :, :] * 255.0).astype(np.uint8) for x in y_train]
    y_train_aug = seq_det.augment_images(y_train_aug)
    y_train_aug = [np.where(x[:, :, :] > 127, 255, 0) for x in y_train_aug]
    y_train_aug = [(x[:, :, :].astype(np.float64)) /
                   255.0 for x in y_train_aug]
    return np.array(X_train_aug), np.array(y_train_aug)


# ### add depth info

# In[54]:


def create_depth_figure(img_size):
    depth_figure = np.zeros((img_size, img_size))
    for i, d in enumerate(np.linspace(0.01, 1, img_size)):
        depth_figure[i, :] = d
    return depth_figure


depth_figure = create_depth_figure(128)


def add_depth_img(img, dfigure):
    shape = list(img.shape)
    shape[-1] = 2
    new_img = np.zeros(shape)
    new_img[:, :, 0] = img[:, :, 0]
    new_img[:, :, 1] = dfigure
    return new_img


def add_depth(img_arr):
    shape = list(img_arr.shape)
    size = shape[1]
    dfigure = create_depth_figure(size)
    return np.array([add_depth_img(x, dfigure) for x in img_arr])


def append_depth_channel(img):
    global depth_figure
    shape = list(img.shape)
    shape[-1] = 2
    new_img = np.zeros(shape)
    new_img[:, :, 0] = img[:, :, 0]
    new_img[:, :, 1] = depth_figure
    return new_img


def append_depth_channel_2(img):
    global depth_figure
    shape = list(img.shape)
    shape[-1] = 3
    new_img = np.zeros(shape)
    new_img[:, :, 0] = img[:, :, 0]
    new_img[:, :, 1] = depth_figure
    new_img[:, :, 2] = depth_figure * img[:, :, 0]
    return new_img


def do_augmentation(seq_det, X_train, y_train):
    X_train_aug = [(x[:, :, :] * 255.0).astype(np.uint8) for x in X_train]
    X_train_aug = seq_det.augment_images(X_train_aug)
    X_train_aug = [append_depth_channel(
        (x[:, :, :].astype(np.float64)) / 255.0) for x in X_train_aug]

    y_train_aug = [(x[:, :, :] * 255.0).astype(np.uint8) for x in y_train]
    y_train_aug = seq_det.augment_images(y_train_aug)
    y_train_aug = [np.where(x[:, :, :] > 120, 255, 0) for x in y_train_aug]
    y_train_aug = [(x[:, :, :].astype(np.float64)) /
                   255.0 for x in y_train_aug]
    return np.array(X_train_aug), np.array(y_train_aug)


def append_augmentation(seq_det, X_train, y_train):
    X_train_aug = [(x[:, :, :] * 255.0).astype(np.uint8) for x in X_train]
    X_train_aug = seq_det.augment_images(X_train_aug)
    X_train_aug = [(x[:, :, :].astype(np.float64)) /
                   255.0 for x in X_train_aug]
    X_train_append = np.append(X_train, X_train_aug, axis=0)
    X_train_append = np.array([append_depth_channel(x)
                               for x in X_train_append])

    y_train_aug = [(x[:, :, :] * 255.0).astype(np.uint8) for x in y_train]
    y_train_aug = seq_det.augment_images(y_train_aug)
    y_train_aug = [np.where(x[:, :, :] > 120, 255, 0) for x in y_train_aug]
    y_train_aug = [(x[:, :, :].astype(np.float64)) /
                   255.0 for x in y_train_aug]
    y_train_append = np.append(y_train, y_train_aug, axis=0)
    return X_train_append, y_train_append


# ### generators

# In[112]:

def generate_batch_data_random(x, y, batch_size, shuffle=True):
    """逐步提取batch数据到显存，降低对显存的占用"""
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
            # seq_det = mixed_seq.to_deterministic()
            # seq_det = intensity_flip_seq.to_deterministic()
            # yield append_augmentation(seq_det, x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])


def generate_batch_data(x, y, batch_size):
    ylen = len(y)
    i = 0
    while 1:
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
        i += 1
        if i * batch_size >= ylen:
            i = 0


def generate_batch_data_random_ref128depth(x, y, batch_size, shuffle=True):
    """逐步提取batch数据到显存，降低对显存的占用"""
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            yield reflect128(add_depth(x[i * batch_size:(i + 1) * batch_size])), reflect128(y[i * batch_size:(i + 1) * batch_size])
            # seq_det = mixed_seq.to_deterministic()
            # seq_det = intensity_flip_seq.to_deterministic()
            # yield append_augmentation(seq_det, x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])


def generate_batch_data_random_ref128depth_aug(x, y, batch_size, shuffle=True, aug_seq=my_seq):
    """逐步提取batch数据到显存，降低对显存的占用"""
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            x_batch = add_depth(x[i * batch_size:(i + 1) * batch_size])
            y_batch = y[i * batch_size:(i + 1) * batch_size]
            seq_det = aug_seq.to_deterministic()
            x_batch, y_batch = my_augmentation(seq_det, x_batch, y_batch)
            yield reflect128(x_batch), reflect128(y_batch)


def generate_batch_data_ref128depth(x, y, batch_size):
    ylen = len(y)
    i = 0
    while 1:
        yield reflect128(add_depth(x[i * batch_size:(i + 1) * batch_size])), reflect128(y[i * batch_size:(i + 1) * batch_size])
        i += 1
        if i * batch_size >= ylen:
            i = 0


def generate_batch_data_random_with_delay(x, y, batch_size, shuffle=True, delay=10):
    """逐步提取batch数据到显存，降低对显存的占用"""
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    epoch = 0
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)

        use_seq = mixed_seq if epoch >= delay else iaa.Noop()

        for i in loop_order:
            seq_det = use_seq.to_deterministic()
            yield do_augmentation(seq_det, x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])

        epoch += 1


def generate256_batch_data_random(x, y, batch_size, shuffle=True):
    """逐步提取batch数据到显存，降低对显存的占用"""
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    seq_det = reshape256_seq.to_deterministic()
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            x_part = append_flip(x[i * batch_size:(i + 1) * batch_size])
            y_part = append_flip(y[i * batch_size:(i + 1) * batch_size])

            yield do_augmentation(seq_det, x_part, y_part)


def generate256_batch_data(x, y, batch_size):
    ylen = len(y)
    seq_det = reshape256_seq.to_deterministic()
    i = 0
    while 1:
        yield do_augmentation(seq_det, x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
        i += 1
        if i * batch_size >= ylen:
            i = 0


def generate128_batch_data_random(x, y, batch_size, shuffle=True):
    """逐步提取batch数据到显存，降低对显存的占用"""
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    seq_det = reshape128_seq.to_deterministic()
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            x_part = append_flip(x[i * batch_size:(i + 1) * batch_size])
            y_part = append_flip(y[i * batch_size:(i + 1) * batch_size])

            yield do_augmentation(seq_det, x_part, y_part)


def generate128_batch_data(x, y, batch_size):
    ylen = len(y)
    seq_det = reshape128_seq.to_deterministic()
    i = 0
    while 1:
        yield do_augmentation(seq_det, x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
        i += 1
        if i * batch_size >= ylen:
            i = 0


# ### 3-task generators

# In[113]:

def get_class_mask(mask):
    cover = (mask > 0.5).sum()
    if cover < 8:
        return 0
    elif np.all(mask == mask[0]):
        return 1
    else:
        return 2


def get_class_labels(mask_arr, classes=3):
    labels = [get_class_mask(mask) for mask in mask_arr]
    arr_len = len(mask_arr)
    one_hot = np.zeros((arr_len, classes))
    one_hot[np.arange(arr_len), labels] = 1
    return one_hot


def generate_batch_data_random_ref128depth_multi(x, y, batch_size, shuffle=True):
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            y_batch = y[i * batch_size:(i + 1) * batch_size]
            y_labels = get_class_labels(y_batch)
            x_batch = reflect128(
                add_depth(x[i * batch_size:(i + 1) * batch_size]))
            y_batch = reflect128(y_batch)
            yield x_batch, {"output_seg": y_batch, "output_fusion": y_batch, "output_clf": y_labels}


def generate_batch_data_random_ref128depth_aug_multi(x, y, batch_size, shuffle=True, aug_seq=my_seq):
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            x_batch = add_depth(x[i * batch_size:(i + 1) * batch_size])
            y_batch = y[i * batch_size:(i + 1) * batch_size]
            y_labels = get_class_labels(y_batch)
            seq_det = aug_seq.to_deterministic()
            x_batch, y_batch = my_augmentation(seq_det, x_batch, y_batch)
            x_batch, y_batch = reflect128(x_batch), reflect128(y_batch)
            yield x_batch, {"output_seg": y_batch, "output_fusion": y_batch, "output_clf": y_labels}


def generate_batch_data_ref128depth_multi(x, y, batch_size):
    ylen = len(y)
    i = 0
    while 1:
        x_batch = add_depth(x[i * batch_size:(i + 1) * batch_size])
        y_batch = y[i * batch_size:(i + 1) * batch_size]
        y_labels = get_class_labels(y_batch)
        x_batch, y_batch = reflect128(x_batch), reflect128(y_batch)
        yield x_batch, {"output_seg": y_batch, "output_fusion": y_batch, "output_clf": y_labels}
        i += 1
        if i * batch_size >= ylen:
            i = 0


# ### deep generator

# In[114]:

reshape064_seq = iaa.Scale({'height': 64, 'width': 64})
reshape032_seq = iaa.Scale({'height': 32, 'width': 32})
reshape016_seq = iaa.Scale({'height': 16, 'width': 16})
reshape008_seq = iaa.Scale({'height': 8, 'width': 8})


def to_uint8(imgs):
    return (imgs * 255).astype(np.uint8)


def to_grey(imgs):
    return imgs.astype(np.float32) / 255.0


def to_mask(imgs):
    return np.where(imgs > 127, 1, 0).astype(np.float32)


def down_cast128(y_imgs):
    global reshape064_seq, reshape032_seq, reshape016_seq, reshape008_seq

    y_imgs = (y_imgs * 255).astype(np.uint8)
    y_imgs064 = reshape064_seq.augment_images(y_imgs)
    y_imgs032 = reshape032_seq.augment_images(y_imgs)
    y_imgs016 = reshape016_seq.augment_images(y_imgs)
    y_imgs008 = reshape008_seq.augment_images(y_imgs)

    return {
        "output_seg064": to_grey(y_imgs064),
        "output_seg032": to_grey(y_imgs032),
        "output_seg016": to_grey(y_imgs016),
        "output_seg008": to_grey(y_imgs008)
    }


def generate_batch_data_random_ref128depth_deep(x, y, batch_size, shuffle=True):
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            y_batch = y[i * batch_size:(i + 1) * batch_size]
            y_labels = get_class_labels(y_batch)
            x_batch = reflect128(add_depth(x[i * batch_size:(i + 1) * batch_size]))
            y_batch = reflect128(y_batch)
            yield x_batch, {"output_seg128": y_batch, "output_fusion": y_batch, "output_clf": y_labels, **down_cast128(y_batch)}


def generate_batch_data_random_ref128depth_aug_deep(x, y, batch_size, shuffle=True, aug_seq=my_seq):
    loop_cnt = div_round(len(y), batch_size)
    loop_order = np.arange(loop_cnt)
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            x_batch = add_depth(x[i * batch_size:(i + 1) * batch_size])
            y_batch = y[i * batch_size:(i + 1) * batch_size]
            y_labels = get_class_labels(y_batch)
            seq_det = aug_seq.to_deterministic()
            x_batch, y_batch = my_augmentation(seq_det, x_batch, y_batch)
            x_batch, y_batch = reflect128(x_batch), reflect128(y_batch)
            yield x_batch, {"output_seg128": y_batch, "output_fusion": y_batch, "output_clf": y_labels, **down_cast128(y_batch)}


def generate_batch_data_ref128depth_deep(x, y, batch_size):
    ylen = len(y)
    i = 0
    while 1:
        x_batch = add_depth(x[i * batch_size:(i + 1) * batch_size])
        y_batch = y[i * batch_size:(i + 1) * batch_size]
        y_labels = get_class_labels(y_batch)
        x_batch, y_batch = reflect128(x_batch), reflect128(y_batch)
        yield x_batch, {"output_seg128": y_batch, "output_fusion": y_batch, "output_clf": y_labels, **down_cast128(y_batch)}
        i += 1
        if i * batch_size >= ylen:
            i = 0


def generate_x_text_ref128depth(x, batch_size=32, flip=False):
    xlen = len(x)
    i = 0
    while 1:
        x_batch = x[i * batch_size:(i + 1) * batch_size]
        if flip:
            x_batch = do_flip(x_batch)
        x_batch = reflect128(add_depth(x_batch))
        yield x_batch
        i += 1
        if i * batch_size >= xlen:
            break


def predict_generator_tta(model, X_valid, batch_size=32):
    steps = div_round(len(X_valid), batch_size)
    X_pred_ori = model.predict_generator(generate_x_text_ref128depth(X_valid, batch_size=batch_size), steps=steps)
    preds_valid_ori = get_101from128(X_pred_ori[0])
    del X_pred_ori
    gc.collect()
    X_pred_flip = model.predict_generator(generate_x_text_ref128depth(X_valid, batch_size=batch_size, flip=True), steps=steps)
    preds_valid_flip = get_101from128(X_pred_flip[0])
    del X_pred_flip
    gc.collect()
    preds_valid_tta = (preds_valid_ori + do_flip(preds_valid_flip)) / 2.0
    print(preds_valid_tta.shape)
    return preds_valid_tta


def predict_generator_notta(model, X_valid, batch_size=32):
    steps = div_round(len(X_valid), batch_size)
    X_pred_ori = model.predict_generator(generate_x_text_ref128depth(X_valid, batch_size=batch_size), steps=steps)
    preds_valid_ori = get_101from128(X_pred_ori[0])
    return preds_valid_ori


def predict_generator(model, X_valid, batch_size=32):
    steps = div_round(len(X_valid), batch_size)
    X_pred_ori = model.predict_generator(generate_x_text_ref128depth(X_valid, batch_size=batch_size), steps=steps)
    return X_pred_ori


def save_deep(base_name, X_pred):
    for i, x in enumerate(X_pred):
        part_name = base_name.replace(".npy", "_{}.npy".format(i))
        print("save", part_name)
        np.save(part_name, x)


def load_deep(base_name):
    prediction = []
    for i in range(7):
        part_name = base_name.replace(".npy", "_{}.npy".format(i))
        print("load", part_name)
        prediction.append(np.load(part_name))
    return prediction


def generate_origin_prediction(x, y_stack, batch_size, shuffle=True):
    loop_cnt = div_round(len(x), batch_size)
    loop_order = np.arange(loop_cnt)
    while 1:
        if shuffle:
            np.random.shuffle(loop_order)
        for i in loop_order:
            x_batch = reflect128(add_depth(x[i * batch_size:(i + 1) * batch_size]))
            y_batch = {
                'output_fusion': y_stack[0][i * batch_size:(i + 1) * batch_size],
                'output_seg128': y_stack[1][i * batch_size:(i + 1) * batch_size],
                'output_seg064': y_stack[2][i * batch_size:(i + 1) * batch_size],
                'output_seg032': y_stack[3][i * batch_size:(i + 1) * batch_size],
                'output_seg016': y_stack[4][i * batch_size:(i + 1) * batch_size],
                'output_seg008': y_stack[5][i * batch_size:(i + 1) * batch_size],
                'output_clf': y_stack[6][i * batch_size:(i + 1) * batch_size]
            }
            yield x_batch, y_batch
