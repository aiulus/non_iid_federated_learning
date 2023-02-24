import numpy as np
import pandas as pd
import math
import statistics
import random
from torchvision import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold


def get_data(path=None):
    mnist_train: datasets = datasets.MNIST(root=path, train=True, download=True, transform=None)
    mnist_test: datasets = datasets.MNIST(root=path, train=False, download=True, transform=None)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets

    return x_train, y_train, x_test, y_test


def sampleWithGaussian(inputData, mu, sig, shuffle=False):
    def calculateSelectionProbs(function, labelSpace, normalization=True):
        # Generate selection probabilities for each label according to given function
        selectionProbs = [function(x) for x in labelSpace]
        if normalization:
            normFactor = 1 / max(selectionProbs)
            selectionProbs = [item * normFactor for item in selectionProbs]
        selectionCriteria = {labelSpace[i]: selectionProbs[i] for i in range(len(labelSpace))}
        return selectionCriteria

    def selectWithProbs(data, selectionCriteria):
        # Select items in dataset according to a dictionary of selection probabilities
        selectedData = []
        for sample in data:
            if random.uniform(0, 1) < float(selectionCriteria.get(sample)):
                selectedData.append(sample)
        return selectedData

    def shuffleNonMax(criteria):
        maxElement = max(criteria, key=criteria.get)
        maxValue = criteria.get(maxElement)
        del criteria[maxElement]
        keys = list(criteria.keys())
        values = list(criteria.values())
        random.shuffle(values)
        newDict = {keys[i]: values[i] for i in range(len(keys))}
        newDict[maxElement] = maxValue
        return newDict

    def gaussian(x, mu, sig):
        return np.exp(- np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * math.sqrt(2 * math.pi))

    # Select 'mu' according to most frequent label
    # sig determines the shape of the curve
    myGaussian = lambda x: gaussian(x, mu=mu, sig=sig)

    # Generate probability vector with myGaussian
    criteria = calculateSelectionProbs(myGaussian, list(set(inputData)))

    # Shuffle the probability vector without effecting the "peak"
    if shuffle: criteria_shuffled = shuffleNonMax(criteria)

    # Select a subset of the original dataset according to the probability vector
    sampledData = selectWithProbs(inputData, criteria)

    return sampledData


def gaussian_mixture(n_clients, n_iter, init_var, var_incr, comp_var):
    gmm = GaussianMixture(n_components=n_clients, covariance_type='full')


def ID_selection(data, id_list):
    result = []
    for ID, x in enumerate(data):
        if ID in id_list:
            result.append(x)
    return result


def partition_homo(dir, n_clients):
    x_train, y_train, x_test, y_test = get_data(dir)
    skf = StratifiedKFold(n_splits=n_clients, shuffle=False)
    subset_indices = []

    for train_index, _ in skf.split(x_train, y_train):
        subset_indices.append(train_index)

    subset_ID_map = {i: subset_indices[i] for i in range(n_clients)}

    return subset_ID_map


def partition_homo_skf(dir, n_clients):
    subset_ID_map = {}
    mnist_train: datasets = datasets.MNIST(root=dir, train=True, download=True, transform=None)
    x_train, y_train = mnist_train.data, mnist_train.targets
    n_train = y_train.shape[0]
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=42)
    subsets = []
    for train_ID, test_ID in skf.split(mnist_train, y_train):
        subsets.append(test_ID)
    for j in range(n_clients):
        # np.random.shuffle(subsets[j])
        subset_ID_map[j] = subsets[j]
    return subset_ID_map


def partition_hetero_dir(dir, n_clients, alpha):
    x_train, y_train, x_test, y_test = get_data(dir)
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

    return subset_ID_map


def partition_hetero_gaussian(dir, n_clients, alpha, bootstrap=False):
    x_train, y_train, x_test, y_test = get_data(dir)
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

    return subset_ID_map


def save_to_csv(subset_map, epoch_num):
    for key in subset_map:
        subset_data = pd.DataFrame([x for index, x in enumerate(x_train.tolist()) if index in subset_map[key]])
        subset_labels = [x for index, x in enumerate(y_train.tolist()) if index in subset_map[key]]
        subset_data['label'] = subset_labels
        subset_data.to_csv(f'subsets/epoch_{epoch_num}_subset_{key + 1}.csv', index=False)


def tensor_to_csv(x_train, y_train, subset_map, epoch_num):
    for key in subset_map:
        x, y = x_train[subset_map.get(key)], y_train[subset_map.get(key)]
        df_x = pd.DataFrame(x.tolist())
        df_y = pd.DataFrame(y.tolist())
        df_x['targets'] = df_y
        df_x.rename(columns={0: 'data', "targets": "targets"})
        df_x.to_csv(f'subsets/epoch_{epoch_num}_subset_{key + 1}.csv', index=False)


def partition(epoch, dir, logdir, type, n_clients, alpha, bootstrap=False, save=False):
    x_train, y_train, x_test, y_test = get_data(dir)
    n_train = x_train.shape[0]

    subset_ID_map = {}

    # Creates a partitioning where each subset has the same distribution of labels as the original dataset
    # via stratified splitting and stores the ID's of data points in each subset in a map.
    if type == "homo":
        subset_ID_map = partition_homo(dir, n_clients)

    # Heterogeneous partitioning by sampling from a Dirichlet process
    # based on https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    elif partition == "hetero-dir":
       subset_ID_map = partition_hetero_dir(dir, n_clients, alpha)

    elif type == "hetero-gaussian":
        subset_ID_map = partition_hetero_gaussian(dir, n_clients, alpha)

    # TODO
    elif type == "hetero-bayesian":
        gmm = GaussianMixture(n_components=n_clients, covariance_type='full')

    return subset_ID_map

def partition(dir, type, n_clients, alpha):
    partition_funcs = {
        "homo" : partition_homo_skf,
        "hetero-dir": partition_hetero_dir,
        "hetero-gaussian": partition_hetero_gaussian
    }
    try:
        partition_func = partition_funcs[type]
    except KeyError:
        raise ValueError(f"Invalid mode: {type}")

    return partition_func(dir, n_clients, alpha)

