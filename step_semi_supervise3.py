from my_dependency import *
from my_metrics import *
from my_builders import *
from my_generators import *

# Set some parameters
exp_code = "035"
img_size_ori = 101
img_size_target = 101
im_width = 128
im_height = 128
im_chan = 1
basicpath = './data/'
path_train = basicpath + 'train/'
path_test = basicpath + 'test/'

path_train_images = path_train + 'images/'
path_train_masks = path_train + 'masks/'
path_test_images = path_test + 'images/'

# Loading of training/testing ids and depths

train_df = pd.read_csv("./data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

print("load data")
train_df["images"] = [np.array(load_img("./data/train/images/{}.png".format(
    idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
train_df["masks"] = [np.array(load_img(
    "./data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
z_cut = pd.cut(train_df.z, [-np.Inf, 100, 200, 300, 400,
                            500, 600, 700, 800, 900, np.Inf], labels=range(10))
c_cut = pd.cut(train_df.coverage,
               [-np.Inf, 0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, np.Inf], labels=range(11))
train_df["c1"] = (train_df.coverage > 0).astype("int")
train_df["z_cut"] = z_cut
train_df["c_cut"] = c_cut

z_group = train_df.groupby(z_cut)
z_cutoff = [-np.Inf, 100, 200, 300, 400, 500, 600, 700, 800, 900, np.Inf]
c_cutoff = [-np.Inf, 0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, np.Inf]
z_cutoff = [-np.Inf, 250, 500, 750, np.Inf]
c_cutoff = [-np.Inf, 0.0, .2, .4, .6, .8, np.Inf]

z_cut = pd.cut(train_df.z, z_cutoff, labels=range(len(z_cutoff) - 1))
c_cut = pd.cut(train_df.coverage, c_cutoff, labels=range(len(c_cutoff) - 1))
train_df["c1"] = (train_df.coverage > 0).astype("int")
train_df["z_cut"] = z_cut
train_df["c_cut"] = c_cut
train_df["cz_cut"] = (c_cut.astype("int") + 1) * 100 + z_cut.astype("int")


# ## 5fold-split --seed 67373

# In[4]:

from sklearn.model_selection import StratifiedKFold


def adjust_split(pair, x_size=3200):
    x, y = pair
    offset = len(x) - x_size
    print(offset)
    if offset > 0:
        y_new = np.append(y, x[:offset])
        x_new = x[offset:].copy()
    elif offset < 0:
        y_new = y[:offset].copy()
        x_new = np.append(x, y[offset:])
    else:
        x_new, y_new = x, y
    return x_new, y_new


X = np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1)
Y = np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1)
y = train_df.cz_cut.values.copy()

skf = StratifiedKFold(n_splits=5, random_state=67373)
skf.get_n_splits(X, y)
kfold_index = [adjust_split(ob) for ob in skf.split(X, y)]

print(y.shape)

for train_idx, valid_idx in kfold_index:
    print(train_idx.shape, valid_idx.shape)


def teacher_name(i):
    return "./035-model-unet3tasks6deepscse-lovasz-stage3-aug-fold-0-run-1.hdf5".replace("stage3", "teacher{}".format(i))


def student_name(i):
    return "./035-model-unet3tasks6deepscse-lovasz-stage3-aug-fold-0-run-1.hdf5".replace("stage3", "stage{}".format(i))


def filter_img(img, th=20):
    if img.sum() < 20:
        return np.zeros(img.shape)
    else:
        return img


def filter_imgs(imgs, th=20):
    return np.array([filter_img(x, th) for x in imgs])


def search_iou(ious, thresholds):
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    return threshold_best, iou_best


def plot_iou(ious, thresholds):
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.show()
    return threshold_best


def do_flip(X):
    return np.array([np.fliplr(x) for x in X])


# In[59]:

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


def prediction_name(i):
    return "./output/035-teacher-{}.npy".format(i)


teacher_stage = 0
teacher_stage = 1
teacher_stage = 2
student_stage = 4
student_stage = 6
student_stage = 7
student_stage = 8
next_student_stage = student_stage + 1
bth = 0.123077

x_test = np.array([(np.array(load_img("./data/test/images/{}.png".format(idx), grayscale=True))) /
                   255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)


# y_test = np.load(prediction_name(teacher_stage))
# y_test = filter_imgs(y_test > bth)
y_test = load_deep(prediction_name(teacher_stage))

run = 1
loss_name = "bcelovasz"

for i, (train_idx, valid_idx) in enumerate(kfold_index):
    keras.backend.clear_session()
    # break
    print("fold:", i)
    print("run:", run)
    print("index.shape:", train_idx.shape, valid_idx.shape)

    pre_model_filepath = student_name(student_stage)
    model_filepath = student_name(next_student_stage)
    log_filepath = student_name(next_student_stage).replace("model", "log").replace(".hdf5", ".csv")

    print("pre:", pre_model_filepath)
    print("mode:", model_filepath)
    print("log:", log_filepath)
    # metric_name = "val_my_iou_metric_2"
    model_params, metric_name = get_model_params("lovasz")
    metric_name = "val_output_fusion_" + metric_name

    print(model_params)
    print(metric_name)

    early_stopping = EarlyStopping(monitor=metric_name, mode='max', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint(model_filepath, monitor=metric_name, mode='max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=metric_name, mode='max', factor=0.5, patience=6, min_lr=0.000005, verbose=1)  # patience=5, factor=0.2
    model_logger = CSVLogger(log_filepath, separator=',', append=False)

    X_valid = X[valid_idx, :]
    Y_valid = Y[valid_idx, :]
    X_train = x_test
    Y_train = y_test
    print("data.shape:", X_train.shape, Y_train[0].shape, X_valid.shape, Y_valid.shape)

    epochs = 50
    train_batch_size = 10
    train_steps_per_epoch = div_round(X_train.shape[0], train_batch_size)

    valid_batch_size = 10
    valid_steps_per_epoch = div_round(X_valid.shape[0], valid_batch_size)

    print("epochs:", epochs)
    print("train_batch_size:", train_batch_size)
    print("train_steps:", train_steps_per_epoch)
    print("train_mult:", train_batch_size * train_steps_per_epoch)

    print("valid_batch_size:", valid_batch_size)
    print("valid_steps:", valid_steps_per_epoch)
    print("valid_mult:", valid_batch_size * valid_steps_per_epoch)
    # break

    # c = Adam(0.005)
    # c = Adam(0.0005)
    # c = SGD(lr=0.001, momentum=.9, decay=1e-3)
    c = Adam(0.0001)
    # c = Adam(0.00005)
    # c = SGD(lr=0.0001, momentum=.9, decay=1e-3)
    model = load_model_bcelovasz2lovasz_deep(pre_model_filepath, c)  # start_neurons = 32
    model = change_loss2mse(model, c)
    print("load model.")

    def train():
        return model.fit_generator(generate_origin_prediction(X_train, Y_train, train_batch_size),
                                   validation_data=generate_batch_data_ref128depth_deep(X_valid, Y_valid, valid_batch_size),
                                   epochs=epochs,
                                   steps_per_epoch=train_steps_per_epoch,
                                   callbacks=[early_stopping, model_checkpoint, model_logger],
                                   validation_steps=valid_steps_per_epoch,
                                   verbose=200)

    train()
    break
