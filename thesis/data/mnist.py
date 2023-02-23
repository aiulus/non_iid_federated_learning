import torch
from torchvision import datasets
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import logging
import math
import random
from sklearn.mixture import GaussianMixture


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


def log_class_counts(y_train, subset_ID_map, logdir):
    cls_counts = {}

    for subset_i, ID in subset_ID_map.items():
        unq, unq_cnt = np.unique(y_train[ID], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        cls_counts[subset_i] = tmp

    logging.debug('Label distributions: %s' % str(cls_counts))

    return cls_counts


def partition(epoch, dir, logdir, type, n_clients, alpha, bootstrap=False, save=False):
    x_train, y_train, x_test, y_test = get_data(dir)
    n_train = x_train.shape[0]

    # Creates a partitioning where each subset has the same distribution of labels as the original dataset
    # via stratified splitting and stores the ID's of data points in each subset in a map.
    if type == "homo":
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

    elif type == "hetero-gaussian":
        min_size = 0
        # classes
        K = 10
        # data points
        N = y_train.shape[0]
        subset_ID_map = {}

        def gaussian_pdf(x, mu, sig):
            return np.exp(- np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * math.sqrt(2 * math.pi))

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
    elif type == "hetero-bayesian":
        gmm = GaussianMixture(n_components=n_clients, covariance_type='full')

    def save_to_csv(subset_map, epoch):
        for client, ID_list in subset_map:
            data = [x for index, x in enumerate(x_train) if index in ID_list]

    if save:
        save_to_csv(subset_ID_map, epoch)

    cls_counts = log_class_counts(y_train, subset_ID_map, logdir)

    return x_train, y_train, x_test, y_test, subset_ID_map, logdir

















