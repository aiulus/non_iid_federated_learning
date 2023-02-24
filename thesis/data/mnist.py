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
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kstest, anderson_ksamp, cumfreq, ks_2samp, cramervonmises, chisquare
from statsmodels.distributions.empirical_distribution import ECDF

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

def map_to_prob(y_train, subset_map):
    counts = log_class_counts(y_train, subset_map)

    values = [np.array([counts.get(k).get(key) for key in counts.get(k)]) for k in counts]
    probs = [values[j] / values[j].sum() for j in range(len(values))]

    return probs

def partition(epoch, path, alpha, mode, n_clients, logpath=None, bootstrap=False):
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

    cls_counts = log_class_counts(y_train, subset_ID_map, logpath)

    return x_train, y_train, x_test, y_test, subset_ID_map, logpath

# TODO: implement method for saving images
def visualize_X(path, epoch, y_train, subset_ID_map, mode_plot, mode_partitioning, alpha, save=False):
    counts = log_class_counts(y_train, subset_ID_map)

    values = [np.array([counts.get(k).get(key) for key in counts.get(k)]) for k in counts]
    values_normalized = [values[j] / values[j].sum() for j in range(len(values))]

    n_clients = len(subset_ID_map)

    title_formats = {
        "hetero-dir": "A distribution-based heterogeneous partitioning X~Dir({alpha}) with {n_clients} subsets",
        "hetero-gaussian": "A distribution-based Gaussian heterogeneous partitioning σ={alpha} with {n_clients} subsets",
        "homo": "A homogeneous partitioning with {n_clients} subsets",
    }
    main_title = title_formats.get(mode_partitioning, "")
    main_title = main_title.format(alpha=alpha, n_clients=n_clients)

    subtitle_formats = {
        "hetero_dir": "α_{epoch}={alpha}",
        "hetero-gaussian": "σ_{epoch}={alpha}",
        "homo" : ""
    }

    subtitle = subtitle_formats.get(mode_partitioning, "")
    subtitle = subtitle.format(epoch=epoch, alpha=alpha)

    if mode_plot == "heatmap":
        ax = sns.heatmap(pd.DataFrame(values_normalized), vmin=0, vmax=1, cmap=sns.cm.rocket_r)
        ax.set(xlabel="Labels", ylabel="Clients", title=main_title)
        plt.show()
    elif mode_plot == "histogram":
        dim_x = 2
        dim_y = 5
        fig, axes = plt.subplots(dim_y, dim_x)
        for j in range(len(values)):
            plt.figure(figsize=(5, 3), dpi=300)
            plt.hist(values[j], [(i - 0.5) / 2 for i in range(20)], label="Sampled dist")
            x = np.arange(-0.5, 9.5, 0.1)
            plt.xticks([i for i in range(10)])
            plt.xlabel("Label")
            plt.xlim([-1, 10])
            plt.ylabel("Entries")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

def visualize(X, Y, xlabels, ylabels, mode, save=False):
    epochs = len(ylabels)

    title_formats = {
        "hetero-dir": "A distribution-based heterogeneous partitioning X~Dir({alpha}) with {n_clients} subsets",
        "hetero-gaussian": "A distribution-based Gaussian heterogeneous partitioning σ={alpha} with {n_clients} subsets",
        "homo": "A homogeneous partitioning with {n_clients} subsets",
    }
    main_title = title_formats.get(mode, "")
    main_title = main_title.format(alpha=alpha, n_clients=n_clients)

    subtitle_formats = {
        "hetero_dir": "α_{epoch}={alpha}",
        "hetero-gaussian": "σ_{epoch}={alpha}",
        "homo" : ""
    }

    subtitle = subtitle_formats.get(mode, "")
    subtitle = subtitle.format(epoch=epoch, alpha=alpha)

    if mode_plot == "heatmap":
        ax = sns.heatmap(pd.DataFrame(values_normalized), vmin=0, vmax=1, cmap=sns.cm.rocket_r)
        ax.set(xlabel="Labels", ylabel="Clients", title=main_title)
        plt.show()
    elif mode_plot == "histogram":
        dim_x = 2
        dim_y = 5
        fig, axes = plt.subplots(dim_y, dim_x)
        for j in range(len(values)):
            plt.figure(figsize=(5, 3), dpi=300)
            plt.hist(values[j], [(i - 0.5) / 2 for i in range(20)], label="Sampled dist")
            x = np.arange(-0.5, 9.5, 0.1)
            plt.xticks([i for i in range(10)])
            plt.xlabel("Label")
            plt.xlim([-1, 10])
            plt.ylabel("Entries")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

def tensor_to_csv(x_train, y_train, subset_map, epoch_num):
    for key in subset_map:
        x, y = x_train[subset_map.get(key)], y_train[subset_map.get(key)]
        df_x = pd.DataFrame(x.tolist())
        df_y = pd.DataFrame(y.tolist())
        df_x['targets'] = df_y
        df_x.rename(columns={0: 'data', "targets": "targets"})
        df_x.to_csv(f'subsets/epoch_{epoch_num}_subset_{key + 1}.csv', index=False)

def distance_X(x_train, y_train, subset_map, mode):
    N = y_train.shape[0]
    probs_original = map_to_prob(y_train, {0: np.array(range(60000))})[0]

    probs = map_to_prob(y_train, subset_map)

    stats = []
    pvals = []

    if mode == "kolmogorov-smirnov":
        for j in range(len(probs)):
            stat, pval = kstest(probs[j], probs_original)
            stats.append(stat)
            pvals.append(pval)
    elif mode == "empirical":
        expected_cdf = ECDF(probs_original)
        for j in range(len(probs)):
            observed_cdf = ECDF(probs[j])
            stat, pval = ks_2samp(observed_cdf(probs[j]), expected_cdf(probs_original))
            stats.append(stat)
            pvals.append(pval)
    elif mode == "cramer-von-mises":
        expected_cdf = ECDF(probs_original)
        for j in range(len(probs)):
            observed_cdf = ECDF(probs[j])
            stat, pval = cramervonmises(observed_cdf, expected_cdf)
            stats.append(stat)
            pvals.append(pval)
    elif mode == "pearson-chi-squared":
        for j in range(len(probs)):
            stat, pval = chisquare(probs[j]*N, probs_original*N)
            stats.append(stat)
            pvals.append(pval)
    # TODO
    elif mode == "anderson-darling":
        for j in range(len(probs)):
            stat, pval = anderson_ksamp(probs[j], probs_original)
            stats.append(stat)
            pvals.append(pval)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return stats, pvals

def distance(y_train, subset_map, mode):
    N = y_train.shape[0]
    probs_original, probs = map_to_prob(y_train, {0: np.arange(60000)})[0], map_to_prob(y_train, subset_map)
    stats, pvals = [], []

    mode_dict = {
        "kolmogorov-smirnov": lambda x: kstest(x, probs_original),
        "empirical": lambda x: ks_2samp(ECDF(x)(x), ECDF(probs_original)(probs_original)),
        "cramer-von-mises": lambda x: cramervonmises(ECDF(x), ECDF(probs_original)),
        "pearson-chi-squared": lambda x: chisquare(x * N, probs_original * N),
        "anderson-darling": anderson_ksamp
    }

    try:
        test_func = mode_dict[mode]
    except KeyError:
        raise ValueError(f"Invalid mode: {mode}")

    stats, pvals = zip(*[test_func(probs[j]) for j in range(len(probs))])

    # for j in range(len(probs)):
    #     stat, pval = test_func(probs[j])
    #     stats.append(stat)
    #     pvals.append(pval)

    return stats, pvals

def run_experiment(alpha_vector, path, logpath, mode, n_clients, save=False):
    x_train, y_train, x_test, y_test = get_data(path)
    evol_stat, evol_pval = [], []
    for j in range(alpha_vector):
        a_j = alpha_vector[j]
        subset_map = partition(j, path, a_j, mode, n_clients)
        if save:
            tensor_to_csv(x_train, y_train, subset_map, j)
        stats, pvals = distance(y_train, subset_map, mode)
        mean_teststat_j = np.mean(stats)
        mean_pval_j = np.mean(pvals)
        evol_stat.append(mean_teststat_j)
        evol_pval.append(mean_pval_j)

