from torch import tensor, from_numpy
import numpy as np
import logging
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kstest, anderson_ksamp, cumfreq, ks_2samp, cramervonmises, chisquare, entropy, \
    wasserstein_distance

import tensorflow as tf
from keras import models, layers, optimizers
from keras.datasets import mnist, cifar10
from keras.layers import Input, Lambda
from keras.layers.preprocessing.normalization import Normalization
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
import visualkeras
from ann_visualizer.visualize import ann_viz

def get_data(dataset_type, path=None):
    if dataset_type == "MNIST":
        train, test = mnist.load_data()
    elif dataset_type == "CIFAR10":
        train, test = cifar10.load_data()

    train_x, train_y = from_numpy(train[0]), from_numpy(train[1])
    test_x, test_y = from_numpy(test[0]), from_numpy(test[1])
    train, test = (train_x, train_y), (test_x, test_y)

    return train, test


def log_class_counts(y_train, subset_ID_map, log=False):
    cls_counts = {}

    for subset_i, ID in subset_ID_map.items():
        unq, unq_cnt = np.unique(y_train[ID], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        cls_counts[subset_i] = tmp

    if log:
        logging.debug('Label distributions: %s' % str(cls_counts))

    return cls_counts


def map_to_prob(y_train, subset_map):
    counts = log_class_counts(y_train, subset_map)

    values = [np.array([counts.get(k).get(key) for key in counts.get(k)]) for k in counts]
    probs = [values[j] / values[j].sum() for j in range(len(values))]

    return probs


def partition_hetero_dir(train, n_clients, alpha):
    x_train, y_train = train[0], train[1]

    min_size = 0
    # classes
    K = len(y_train.unique())
    # data points
    N = y_train.shape[0]
    subset_ID_map = {}

    while min_size < 10:
        subset_ID_list = [[] for _ in range(n_clients)]
        for k in range(K):
            ids_k = np.where(y_train == k)[0]
            np.random.shuffle(ids_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = np.array(
                [p * (len(ids_j) < N / n_clients) for p, ids_j in zip(proportions, subset_ID_list)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(ids_k)).astype(int)[:-1]
            subset_ID_list = [ids_j + ids.tolist() for ids_j, ids in
                              zip(subset_ID_list, np.split(ids_k, proportions))]
            min_size = min([len(ids_j) for ids_j in subset_ID_list])

    for j in range(n_clients):
        np.random.shuffle(subset_ID_list[j])
        subset_ID_map[j] = subset_ID_list[j]

    return subset_ID_map


def partition(dataset_type, type, n_clients, alpha):
    train, test = get_data(dataset_type, "D:/datasets")

    partition_funcs = {
        # "homo": partition_homo_skf,
        "hetero-dir": partition_hetero_dir,
        # "hetero-gaussian": partition_hetero_gaussian,
        # "quant": partition_quantity_based
    }

    try:
        partition_func = partition_funcs[type]
    except KeyError:
        raise ValueError(f"Invalid mode: {type}")

    return partition_func(train, n_clients, alpha)


def get_layers_functions(dataset_type, model_type):
    cnn_layers = {
        "MNIST": [Conv2D(filters=64,
                         kernel_size=3,
                         activation='relu',
                         input_shape=(28, 28, 1)),
                  MaxPooling2D(2),
                  Conv2D(filters=64,
                         kernel_size=3,
                         activation='relu'),
                  MaxPooling2D(2),
                  Conv2D(filters=64,
                         kernel_size=3,
                         activation='relu'),
                  Flatten(),
                  Dropout(0.2),
                  Dense(10, activation='softmax')
                  ],
        "CIFAR10": [Conv2D(32, (3, 3),
                           activation='relu',
                           input_shape=(32, 32, 3)),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3),
                           activation='relu'),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3),
                           activation='relu'),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(10)]
    }

    cnn_loss = {
        "MNIST": CategoricalCrossentropy(),
        "CIFAR10": SparseCategoricalCrossentropy(from_logits=True)
    }

    # TODO: Find best configuration for CIFAR10
    nn_layers = {
        "MNIST": [
            Flatten(input_shape=(28, 28)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dropout(rate=0.4),
            Dense(units=10, activation='softmax')
        ],
        "CIFAR10": [
            Flatten(input_shape=(28, 28)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dropout(rate=0.4),
            Dense(units=10, activation='softmax')
        ]
    }

    nn_loss = {
        "MNIST": SparseCategoricalCrossentropy(),
        "CIFAR10": SparseCategoricalCrossentropy()
    }

    if model_type == "nn":
        return nn_layers.get(dataset_type), nn_loss.get(dataset_type)
    elif model_type == "cnn":
        return cnn_layers.get(dataset_type), cnn_loss.get(dataset_type)
    else:
        raise ValueError(f"Invalid mode: {model_type}")


def build_model(dataset_type, model_type, data, n_epochs, lrate, batch_size=None, plot=False):
    train_data, train_targets, test_data, test_targets = data

    layers, loss_function = get_layers_functions(dataset_type, model_type)

    model = Sequential()

    for layer in layers:
        model.add(layer)

    model.compile(loss=loss_function,
                  optimizer=optimizers.Adam(learning_rate=lrate),
                  metrics=['accuracy'])

    history = model.fit(train_data, train_targets,
                        epochs=n_epochs,
                        validation_data=(test_data, test_targets))

    if plot:
        filename = dataset_type + "_" + model_type + "_epochs_" + f'{n_epochs}' + ".png"
        plot_model(model, to_file="keras-utils_"+filename+".png",
                   show_shapes=True, show_layer_names=True, show_layer_activations=True)
        ann_viz(model, view=True, filename="annviz_"+filename)
        visualkeras.layered_view(model, to_file="visualkeras_" + filename + ".png")

    loss, accuracy = model.evaluate(test_data, test_targets, batch_size=batch_size)

    return accuracy, loss, history.epoch, pd.DataFrame(history.history)

def normalized_euclidian_transform():
    (cifar_train_data, cifar_train_targets), (cifar_test_data, cifar_test_targets) = cifar10.load_data()

    norm_layer = Normalization()
    norm_layer.adapt(cifar_train_data)
    cifar_train_data = norm_layer(cifar_train_data)
    cifar_test_data = norm_layer(cifar_test_data)

    input_shape = cifar_train_data.shape[1:]
    inputs = Input(shape=input_shape)

    distance = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x-0.5), axis=-1)))(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=distance)

    cifar_train_data = model.predict(cifar_train_data)
    cifar_train_targets = np.squeeze(cifar_train_targets, axis=-1)
    cifar_test_data = model.predict(cifar_test_data)
    cifar_test_targets = np.squeeze(cifar_test_targets, axis=-1)

    return (cifar_train_data, cifar_train_targets), (cifar_test_data, cifar_test_targets)


def run_experiment(model_type, dataset_type, partitioning_type, n_clients, alpha, n_epochs, lrate=0.003,
                   batch_size=None):
    dataset_loaders = {
        "MNIST": mnist.load_data,
        "CIFAR10": normalized_euclidian_transform,
    }

    try:
        loader = dataset_loaders[dataset_type]
    except KeyError:
        raise ValueError(f"Invalid mode: {dataset_type}")

    train, test = loader()
    test_data, test_targets = test[0], test[1]

    subset_map = partition(dataset_type, partitioning_type, n_clients, alpha)
    accuracies = []

    for j in range(len(subset_map)):
        subset_j_data = train[0][subset_map[j]]
        subset_j_targets = train[1][subset_map[j]]

        # convert to one-hot vector
        subset_j_targets = to_categorical(subset_j_targets, num_classes=10)
        test_targets = to_categorical(test_targets, num_classes=10)

        subset_j_data = subset_j_data.reshape((-1,) + subset_j_data.shape[1:3] + (1, ))
        test_data = test_data.reshape((-1,) + test_data.shape[1:3] + (1, ))

        if model_type == "nn":
            subset_j_data = subset_j_data.astype('float32')/255
            test_data = test_data.astype('float')/255

        data_j = (subset_j_data, subset_j_targets, test_data, test_targets)
        # accuracy, loss, epochs, hist = build_model(dataset_type, model_type, data_j, n_epochs, lrate)
        accuracy = build_model(dataset_type, model_type, data_j, n_epochs, lrate)[0]

        accuracies.append(accuracy)

    return accuracies

