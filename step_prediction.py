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


X = np.array(train_df.images.tolist()).reshape(-1,
                                               img_size_target, img_size_target, 1)
Y = np.array(train_df.masks.tolist()).reshape(-1,
                                              img_size_target, img_size_target, 1)
y = train_df.cz_cut.values.copy()

skf = StratifiedKFold(n_splits=5, random_state=67373)
skf.get_n_splits(X, y)
kfold_index = [adjust_split(ob) for ob in skf.split(X, y)]

print(y.shape)

for train_idx, valid_idx in kfold_index:
    print(train_idx.shape, valid_idx.shape)


def teacher_name(i):
    return "./035-model-unet3tasks6deepscse-lovasz-stage3-aug-fold-0-run-1.hdf5".replace("stage3", "teacher{}".format(i))


# ## serach stage

# In[140]:

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


teacher_stage = 0
# teacher_stage = 1
# teacher_stage = 2

model_name = teacher_name(teacher_stage)
print(model_name)
teacher_model = load_model_with_allco(model_name)
print("load")
model = teacher_model
thresholds = np.linspace(-0.8, 1.2, 40)

for i, (train_idx, valid_idx) in enumerate(kfold_index):
    X_valid = X[valid_idx, :]
    Y_valid = Y[valid_idx, :]

    X_valid_flip = reflect128(do_flip(add_depth(X_valid)))
    X_valid_ori = reflect128(add_depth(X_valid))
    print(X_valid_ori.shape)
    X_pred_ori = model.predict(X_valid_ori, batch_size=32)
    X_pred_flip = model.predict(X_valid_flip, batch_size=32)
    preds_valid_ori = get_101from128(X_pred_ori[0])
    del X_pred_ori
    gc.collect()
    preds_valid_flip = get_101from128(X_pred_flip[0])
    del X_pred_flip
    gc.collect()
    preds_valid_tta = (preds_valid_ori + do_flip(preds_valid_flip)) / 2.0
    del preds_valid_flip
    gc.collect()
    print(preds_valid_tta.shape)

    print("no filter")
    preds_valid = preds_valid_ori
    ious = np.array([get_iou_vector(Y_valid.reshape((-1, img_size_ori, img_size_ori)),
                                    preds_valid.reshape((-1, img_size_ori, img_size_ori)) > threshold) for threshold in tqdm(thresholds)])
    bth, biou = search_iou(ious, thresholds)
    print(bth, biou)

    print("use filter")
    ious = np.array([get_iou_vector(Y_valid.reshape((-1, img_size_ori, img_size_ori)), filter_imgs(
        preds_valid.reshape((-1, img_size_ori, img_size_ori)) > threshold, 20)) for threshold in tqdm(thresholds)])
    bth, biou = search_iou(ious, thresholds)
    print(bth, biou)

    print("no filter, tta")
    preds_valid = preds_valid_tta
    ious = np.array([get_iou_vector(Y_valid.reshape((-1, img_size_ori, img_size_ori)),
                                    preds_valid.reshape((-1, img_size_ori, img_size_ori)) > threshold) for threshold in tqdm(thresholds)])
    bth, biou = search_iou(ious, thresholds)
    print(bth, biou)

    print("use filter, tta")
    ious = np.array([get_iou_vector(Y_valid.reshape((-1, img_size_ori, img_size_ori)), filter_imgs(
        preds_valid.reshape((-1, img_size_ori, img_size_ori)) > threshold, 20)) for threshold in tqdm(thresholds)])
    bth, biou = search_iou(ious, thresholds)
    print(bth, biou)
    break

del X_valid, Y_valid, X_valid_flip, X_valid_ori
del preds_valid_ori, preds_valid_tta
del train_df
del X, Y
gc.collect()

print("search done")
x_test = np.array([(np.array(load_img("./data/test/images/{}.png".format(idx), grayscale=True))) /
                   255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)


def prediction_name(i):
    return "./output/035-teacher-{}.npy".format(i)


prediction = predict_generator_tta(model, x_test)
print(prediction.shape)
np.save(prediction_name(teacher_stage), prediction)
