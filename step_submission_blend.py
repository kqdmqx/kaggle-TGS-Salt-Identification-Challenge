from my_dependency import *
from my_metrics import *
from my_builders import *
from my_generators import *


def teacher_name(i):
    return "./035-model-unet3tasks6deepscse-lovasz-stage3-aug-fold-0-run-1.hdf5".replace("stage3", "teacher{}".format(i))


def prediction_name_sub(i):
    return "./output/035-blend-{}.npy".format(i)


def submission_name(model_name):
    return model_name.replace("./output/", "./result/").replace(".npy", ".csv")


def prediction_name(i):
    return "./output/035-{}.npy".format(i)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def filter_img(img, th=20):
    if img.sum() < 20:
        return np.zeros(img.shape)
    else:
        return img


def filter_imgs(imgs, th=20):
    return np.array([filter_img(x, th) for x in imgs])


train_df = pd.read_csv("./data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

teacher_stage = 2

prediction_list = (
    ("student-10", -0.0308, .1),
    ("student-9", 0.1744, .1),
    ("student-8", 0.1744, .05),
    ("student-7", -0.1846, .1),
    ("student-6", 0.1744, .1),
    ("student-5", -0.3897, 1.0),
    ("teacher-2", 0.2769, 1.0),
    ("teacher-1", 0.1231, 2.0),
    ("teacher-0", -0.0821, .1),
)

# total_weight = 0
# for i, (stage, th, weight) in enumerate(prediction_list):
#     prediction_path = prediction_name(stage)
#     print("load:", prediction_path)
#     if i == 0:
#         y_test = sigmoid(np.load(prediction_path) - th) * weight
#     else:
#         y_test += sigmoid(np.load(prediction_path) - th) * weight
#     total_weight += weight

# y_test = y_test / total_weight
# np.save("./output/035-teacher-3.npy", y_test)
# y_test = filter_imgs(y_test > 0.5)
y_test = np.load("./output/035-teacher-3.npy")
y_test = y_test > 0.5

import time
t1 = time.time()
pred_dict = {idx: rle_encode2(y_test[i])
             for i, idx in enumerate(tqdm(test_df.index.values))}
t2 = time.time()
print("cost", t2 - t1, "s")

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
print(submission_name(prediction_name_sub(teacher_stage)))
print(sub.head(10))
sub.to_csv(submission_name(prediction_name_sub(teacher_stage)))
