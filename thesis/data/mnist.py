import torch
from torchvision import datasets
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import logging
import math
import random
from sklearn.mixture import GaussianMixture
import pandas as pd


def get_data(path=None):
    mnist_train: datasets = datasets.MNIST(root=path, train=True, download=True, transform=None)
    mnist_test: datasets = datasets.MNIST(root=path, train=False, download=True, transform=None)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets

    return x_train, y_train, x_test, y_test


# function that returns a map which stores the indices of data points belonging to each label (0:9)
def parse_class_distr(data):
    class_ID_map = {}

    for ID, classes in enumerate(data):
        for cls in classes:
            if cls not in class_ID_map:
                class_ID_map[cls] = []
            class_ID_map[cls].append(ID)


def log_class_counts(y_train, subset_ID_map, log=False):
    cls_counts = {}

    for subset_i, ID in subset_ID_map.items():
        unq, unq_cnt = np.unique(y_train[ID], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        cls_counts[subset_i] = tmp

    if log:
        logging.debug('Label distributions: %s' % str(cls_counts))

    return cls_counts


def partition(epoch, path, logpath, mode, n_clients, alpha, bootstrap=False, save=False):
    x_train, y_train, x_test, y_test = get_data(path)
    n_train = x_train.shape[0]

    # Creates a partitioning where each subset has the same distribution of labels as the original dataset
    # via stratified splitting and stores the ID's of data points in each subset in a map.
    if mode == "homo":
        skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=42)
        subset_indices = []

        for train_ID, test_ID in skf.split(x_train, y_train):
            subset_indices.append(test_ID)

        subset_ID_map = {i: subset_indices[i] for i in range(n_clients)}

    # Heterogeneous partitioning by sampling from a Dirichlet process
    # based on https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    elif partition == "hetero-dir":
        min_size = 0
        # classes
        K = 10
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

    elif mode == "hetero-gaussian":
        min_size = 0
        # classes
        K = 10
        # data points
        N = y_train.shape[0]
        subset_ID_map = {}

        def gaussian_pdf(x, mean, variance):
            return np.exp(- np.power(x - mean, 2.) / (2 * np.power(variance, 2.))) / (variance * math.sqrt(2 * math.pi))

        labels = list(set(y_train.tolist()))
        mu = 5
        sig = alpha
        proportions = np.array([gaussian_pdf(x, mu, sig) for x in labels])

        if bootstrap:
            # make sure the most frequent label always gets selected
            norm_const = 1 / max(proportions)
            proportions = [item * norm_const for item in proportions]
            prob_map = {labels[i]: proportions[i] for i in range(len(labels))}

            def shuffle_labels(map):
                max_item = max(map, key=map.get)
                max_val = map.get(max_item)
                del map[max_item]
                keys = list(map.keys())
                values = list(map.values())
                random.shuffle(values)
                new_map = {keys[i]: values[i] for i in range(len(keys))}
                new_map.update({max_item: max_val})
                return new_map

            prob_map = shuffle_labels(prob_map)

            subset_ID_list = [[] for _ in range(n_clients)]

            for k in range(K):
                sample = []

                for ID, x in enumerate(y_train.tolist):
                    if random.uniform(0, 1) < float(prob_map.get(x)):
                        sample.append(ID)

                subset_ID_list[k] = sample

        else:
            while min_size < 10:
                subset_ID_list = [[] for _ in range(n_clients)]
                for k in range(K):
                    ids_k = np.where(y_train == k)[0]
                    np.random.shuffle(ids_k)
                    proportions = np.array([gaussian_pdf(x, mu, sig) for x in labels])
                    proportions = np.array(
                        [p * (len(ids_j) < N / n_clients) for p, ids_j in zip(proportions, subset_ID_list)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(ids_k)).astype(int)[:-1]
                    subset_ID_list = [ids_j + ids.tolist() for ids_j, ids in
                                      zip(subset_ID_list, np.split(ids_k, proportions))]
                    min_size = min([len(ids_j) for ids_j in subset_ID_list])

        subset_ID_map = {i: subset_ID_list[i] for i in range(n_clients)}

    # TODO
    elif mode == "hetero-bayesian":
        gmm = GaussianMixture(n_components=n_clients, covariance_type='full')

    def save_to_csv(subset_map, epoch_num):
        for key in subset_map:
            subset_data = pd.DataFrame([x for index, x in enumerate(x_train.tolist()) if index in subset_map[key]])
            subset_labels = [x for index, x in enumerate(y_train.tolist()) if index in subset_map[key]]
            subset_data['label'] = subset_labels
            subset_data.to_csv(f'epoch_{epoch_num}_subset_{key + 1}.csv', index=False)

    if save:
        save_to_csv(subset_ID_map, epoch)

    cls_counts = log_class_counts(y_train, subset_ID_map, logpath)

    return x_train, y_train, x_test, y_test, subset_ID_map, logpath

def visualize(path, epoch, y_train, subset_ID_map, mode_plot, mode_partitioning, alpha, save=False):
    counts = log_class_counts(y_train, subset_ID_map)

    values = np.array([np.array([counts.get(k).get(key) for key in counts.get(k)]) for k in counts])
    values_normalized = [values[j] / values[j].sum() for j in range(len(values))]

    n_clients = len(subset_ID_map)

    title_formats = {
        "hetero-dir": "A distribution-based heterogeneous partitioning X~Dir({alpha}) with {n_clients} subsets",
        "hetero-gaussian": "A distribution-based Gaussian heterogeneous partitioning σ={alpha} with {n_clients} subsets",
        "homo": "A homogeneous partitioning with {n_clients} subsets",
    }
    main_title = title_formats.get(mode_partitioning, "")
    main_title = main_title.format(alpha=alpha, n_clients=n_clients)

    if mode_plot == "heatmap":











