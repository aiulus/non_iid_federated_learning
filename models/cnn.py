import experiments.mnist_experiment as data
from torchvision import datasets
from keras.preprocessing.image import ImageDataGenerator
from torch import tensor
import pymc3 as py

x_train, y_train, x_test, y_test = data.get_data("C:/Users/Aybüke/PycharmProjects/datasets")
subset_IDs = data.partition("C:/Users/Aybüke/PycharmProjects/datasets", "hetero-dir", 10, 0.5)

x_subsets = [x_train[subset_IDs.get(key)] for key in range(len(subset_IDs))]
y_subsets = [y_train[subset_IDs.get(key)] for key in range(len(subset_IDs))]

datagen = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25)
datagen.fit(x_subsets[0].reshape(x_subsets[0].shape[0], 28, 28, 1))
extra = 10
shift_0_iterator = datagen.flow(x_subsets[0].reshape(x_subsets[0].shape[0], 28, 28, 1),
                                y_subsets[0].reshape(y_subsets[0].shape[0], 1), batch_size=extra, shuffle=False)
x_subsets_0 = [x for x, y in shift_0_iterator]
y_subsets_0 = [y for x, y in shift_0_iterator]


def augment(method, x_subset, y_subset, batch_size):
    rotation_range_val = 30
    width_shift_range = 0.25
    height_shift_range = 0.25
    shear_range_val = 45
    zoom_range_val = [0.5, 1.5]
    methods = {
        "rotate": ImageDataGenerator(rotation_range=rotation_range_val),
        "shift": ImageDataGenerator(width_shift_range=width_shift_range, height_shift_range=height_shift_range),
        "shear": ImageDataGenerator(shear_range=shear_range_val),
        "zoom": ImageDataGenerator(zoom_range=zoom_range_val)
    }

    data_gen = methods[method]
    data_gen.fit(x_subset.reshape(x_subset.shape[0], 28, 28, 1))
    iterator = data_gen.flow(x_subset.reshape(x_subset.shape[0], 28, 28, 1), y_subset.reshape(y_subset.shape[0], 1), batch_size=batch_size, shuffle=False)
    
    append_X = []
    append_Y = []
    
    for x_subset, y_subset in iterator:
        x_subset = x_subset.reshape(x_subset.shape[0], 28, 28)
        y_subset = y_subset.reshape(y_subset.shape[0])
        append_X.append(x_subset)
        append_Y.append(y_subset)

    t_X = tensor(append_X)
    t_Y = tensor(append_Y)

    x_augmented = tensor([t_X, x_subset])
    y_augmented = tensor([t_Y, y_subset])

    return x_augmented, y_augmented