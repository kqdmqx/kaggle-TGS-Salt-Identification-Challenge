from my_dependency import *
from my_metrics import *
from my_builders import *
from my_generators import *


def update_weights(ow, nw, alpha):
    return ow * (1 - alpha) + nw * alpha


def update_layer(layer_old, layer_new, alpha):
    old_weights = layer_old.get_weights()
    new_weights = layer_new.get_weights()
    if isinstance(old_weights, tuple) or isinstance(old_weights, list):
        new_weights = [update_weights(ow, nw, alpha) for ow, nw in zip(old_weights, new_weights)]
    else:
        new_weights = update_weights(old_weights, new_weights, alpha)
    layer_old.set_weights(new_weights)


def update_model(model_old, model_new, alpha):
    for i, layer in tqdm(enumerate(model_old.layers)):
        update_layer(layer, model_new.layers[i], alpha)


def teacher_name(i):
    return "./035-model-unet3tasks6deepscse-lovasz-stage3-aug-fold-0-run-1.hdf5".replace("stage3", "teacher{}".format(i))


def student_name(i):
    return "./035-model-unet3tasks6deepscse-lovasz-stage3-aug-fold-0-run-1.hdf5".replace("stage3", "stage{}".format(i))


teacher_stage = 0
teacher_stage = 1
next_teacher_stage = teacher_stage + 1
student_stage = 5
student_stage = 7
next_student_stage = student_stage + 1

print("from: ", teacher_name(teacher_stage))
print("to: ", teacher_name(next_teacher_stage))

print("update1: ", student_name(student_stage))
print("update2: ", student_name(next_student_stage))


teacher_model = load_model_with_allco(teacher_name(teacher_stage))
print("susccess load:", teacher_name(teacher_stage))
student_model_1 = load_model_with_allco(student_name(student_stage))
print("susccess load:", student_name(student_stage))
student_model_2 = load_model_with_allco(student_name(next_student_stage))
print("susccess load:", student_name(next_student_stage))

print(teacher_model.layers[2].get_weights())
update_model(teacher_model, student_model_1, 0.25)
print("susccess update with:", student_name(student_stage))
print(teacher_model.layers[2].get_weights())
update_model(teacher_model, student_model_2, 0.25)
print("susccess update with:", student_name(next_student_stage))
print(teacher_model.layers[2].get_weights())

teacher_model.save(teacher_name(next_teacher_stage))
