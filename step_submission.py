from my_dependency import *
from my_metrics import *
from my_builders import *
from my_generators import *


def teacher_name(i):
    return "./035-model-unet3tasks6deepscse-lovasz-stage3-aug-fold-0-run-1.hdf5".replace("stage3", "teacher{}".format(i))


def submission_name(model_name):
    return model_name.replace("./output/", "./result/").replace(".npy", ".csv")


def prediction_name(i):
    return "./output/035-teacher-{}.npy".format(i)


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

teacher_stage = 0
teacher_stage = 1
student_stage = 4
student_stage = 6
next_student_stage = student_stage + 1
bth = 0.123077

y_test = np.load(prediction_name(teacher_stage))
y_test = filter_imgs(y_test > bth)

import time
t1 = time.time()
pred_dict = {idx: rle_encode2(y_test[i]) for i, idx in enumerate(tqdm(test_df.index.values))}
t2 = time.time()
print("cost", t2 - t1, "s")

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
print(submission_name(prediction_name(teacher_stage)))
print(sub.head(10))
sub.to_csv(submission_name(prediction_name(teacher_stage)))
