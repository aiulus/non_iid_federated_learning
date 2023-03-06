#!/usr/bin/env python
# coding: utf-8

# In[92]:


import torch
from torchvision import datasets, transforms
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
import logging
import math
import random
from sklearn.mixture import GaussianMixture
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kstest, anderson_ksamp, cumfreq, ks_2samp, cramervonmises, chisquare, entropy
from tensorflow import Tensor

# from statsmodels.distributions.empirical_distribution import ECDF


# In[6]:


def get_data(dataset_type, path=None):
    if dataset_type == "MNIST":
        train: datasets = datasets.MNIST(root=path, train=True, download=True, transform=None)
        test: datasets = datasets.MNIST(root=path, train=False, download=True, transform=None)
    elif dataset_type == "CIFAR10":
        train: datasets = datasets.CIFAR10(root=path, train=True, download=True, transform=transforms.ToTensor)
        test: datasets = datasets.CIFAR10(root=path, train=False, download=True, transform=transforms.ToTensor)

    x_train, y_train = train.data, train.targets
    x_test, y_test = test.data, test.targets

    return x_train, y_train, x_test, y_test


# In[7]:


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


# In[8]:


def log_class_counts(y_train, subset_ID_map, log=False):
    cls_counts = {}

    for subset_i, ID in subset_ID_map.items():
        unq, unq_cnt = np.unique(y_train[ID], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        cls_counts[subset_i] = tmp

    if log:
        logging.debug('Label distributions: %s' % str(cls_counts))

    return cls_counts


# In[9]:


def map_to_prob(y_train, subset_map):
    counts = log_class_counts(y_train, subset_map)

    values = [np.array([counts.get(k).get(key) for key in counts.get(k)]) for k in counts]
    probs = [values[j] / values[j].sum() for j in range(len(values))]

    return probs


# In[10]:


def partition_homo_skf(dir, n_clients, alpha=0):
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

    # TODO: might make more sense to return the original dataset as well
    return subset_ID_map


# In[11]:


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

    # TODO: might make more sense to return the original dataset as well
    return subset_ID_map


# In[12]:

dataset_type = "MNIST"
dataset = getattr(datasets, dataset_type)

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

    # TODO: might make more sense to return the original dataset as well
    return subset_ID_map


# In[13]:


def sorted_minibatches(path):
    train = datasets.MNIST(root=path, train=True, download=True, transform=None)

    x, y = train.data, np.array(train.targets)

    subsets_pure = {i: np.where(y == i)[0] for i in range(10)}

    return subsets_pure


def quantity(path, n_clients, alpha):
    # sorted = pd.concat([y for x, y in data.groupby(0)]).reset_index().drop(columns=['index'])
    train = datasets.MNIST(root=path, train=True, download=True, transform=None)

    x, y = train.data, np.array(train.targets)

    # subsets_pure = {i: np.where(y == i)[0] for i in range(10)}
    subsets_pure = np.concatenate([np.where(y == i)[0] for i in range(10)])

    ids = np.arange(train.targets.shape[0])
    batch_ids = np.array_split(ids, n_clients*alpha)
    minibatches = []

    for i in range(n_clients*alpha):
        batch_i = [subsets_pure[j] for j in batch_ids[i]]
        minibatches.append(batch_i)

    minibatches = np.array(minibatches)

    # Generate m random sets and assign them to clients:

    ids_mini = np.random.permutation(range(n_clients*alpha))
    subset_indices = np.array_split(ids_mini, n_clients)

    clients = {}

    for index in subset_indices:
        client_i = []
        for i in index:
            client_i.append(minibatches[i])
        client_i = np.concatenate(client_i)
        clients.update({i:np.array(client_i)})

    return clients


def quantity_based_hetero(data, n_clients, alpha):
    minibatches = sorted_minibatches(data, n_clients, alpha)

    # Generate m random sets and assign them to clients:

    ids_mini = np.random.permutation(range(n_clients * alpha))
    subset_indices = np.array_split(ids_mini, n_clients)

    clients = {}

    for index in subset_indices:
        client_i = []
        for i in index:
            client_i.append(minibatches[i])

        client_i = np.concatenate(client_i)
        clients.update({i: np.array(client_i)})

    return np.array(clients)


def partition_quantity_based(path, n_clients, alpha):
    train = datasets.MNIST(root=path, train=True, download=True, transform=None)
    x, y = train.data, np.array(train.targets)
    subsets_pure = {i: np.where(y == i)[0] for i in range(10)}

    ids_mini = np.random.permutation(range(n_clients * alpha))
    subset_indices = np.array_split(ids_mini, n_clients)

    subset_map = {i: np.concatenate([subsets_pure.get(j) for j in index]) for i, index in enumerate(subset_indices)}

    return subset_map

def partition_quantity_based(path, n_clients, alpha):
    # sorted = pd.concat([y for x, y in data.groupby(0)]).reset_index().drop(columns=['index'])
    train = datasets.MNIST(root=path, train=True, download=True, transform=None)

    x, y = train.data, np.array(train.targets)

    # subsets_pure = {i: np.where(y == i)[0] for i in range(10)}
    subsets_pure = np.concatenate([np.where(y == i)[0] for i in range(10)])

    ids = np.arange(train.targets.shape[0])
    batch_ids = np.array_split(ids, n_clients*alpha)
    minibatches = []

    for i in range(n_clients*alpha):
        batch_i = [subsets_pure[j] for j in batch_ids[i]]
        minibatches.append(batch_i)

    minibatches = np.array(minibatches)

    # Generate m random sets and assign them to clients:

    ids_mini = np.random.permutation(range(n_clients*alpha))
    subset_indices = np.array_split(ids_mini, n_clients)

    clients = {}

    for index in subset_indices:
        client_i = []
        for i in index:
            client_i.append(minibatches[i])
        client_i = np.concatenate(client_i)
        clients.update({i:np.array(client_i)})

    return clients


def partition(dir, type, n_clients, alpha):
    partition_funcs = {
        "homo": partition_homo_skf,
        "hetero-dir": partition_hetero_dir,
        "hetero-gaussian": partition_hetero_gaussian
    }

    try:
        partition_func = partition_funcs[type]
    except KeyError:
        raise ValueError(f"Invalid mode: {type}")

    return partition_func(dir, n_clients, alpha)


# In[14]:


def visualize(path, epoch, y_train, subset_ID_map, mode_plot, mode_partitioning, alpha, save=False):
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
        "homo": ""
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


# In[15]:


def tensor_to_csv(x_train, y_train, subset_map, epoch_num):
    for key in subset_map:
        x, y = x_train[subset_map.get(key)], y_train[subset_map.get(key)]
        df_x = pd.DataFrame(x.tolist())
        df_y = pd.DataFrame(y.tolist())
        df_x['targets'] = df_y
        df_x.rename(columns={0: 'data', "targets": "targets"})
        df_x.to_csv(f'subsets/epoch_{epoch_num}_subset_{key + 1}.csv', index=False)


# In[16]:


x_train, y_train, x_test, y_test = get_data("C:/Users/Aybüke/PycharmProjects/datasets")

# In[17]:


map_homo = partition("C:/Users/Aybüke/PycharmProjects/datasets", "homo", 10, 0)
map_dir_05 = partition("C:/Users/Aybüke/PycharmProjects/datasets", "hetero-dir", 10, 0.5)
map_dir_1000 = partition("C:/Users/Aybüke/PycharmProjects/datasets", "hetero-dir", 10, 1000)
map_gaussian_1 = partition("C:/Users/Aybüke/PycharmProjects/datasets", "hetero-gaussian", 10, 1)
map_gaussian_20 = partition("C:/Users/Aybüke/PycharmProjects/datasets", "hetero-gaussian", 10, 20)
map_gaussian_1000 = partition("C:/Users/Aybüke/PycharmProjects/datasets", "hetero-gaussian", 10, 1000)

# In[18]:


count_homo = log_class_counts(y_train, map_homo)
count_dir_05 = log_class_counts(y_train, map_dir_05)
count_dir_1000 = log_class_counts(y_train, map_dir_1000)
count_gaussian_1 = log_class_counts(y_train, map_gaussian_1)
count_gaussian_20 = log_class_counts(y_train, map_gaussian_20)
count_gaussian_1000 = log_class_counts(y_train, map_gaussian_1000)

# In[19]:


values = [np.array([count_homo.get(k).get(key) for key in count_homo.get(k)]) for k in count_homo]
counts_normalized = [values[j] / values[j].sum() for j in range(len(values))]
counts_normalized

# In[20]:


# homogenous
fig, axes = plt.subplots(5, 2)
for j in range(len(counts_normalized)):
    ax = sns.lineplot(counts_normalized[j], ax=axes[int(j / 2), j % 2])
    ax.set(xlabel="Label Frequency", ylabel="Clients")
plt.show()

# In[21]:


# heterogenous
values = [np.array([count_dir_05.get(k).get(key) for key in count_dir_05.get(k)]) for k in count_dir_05]
counts_normalized = [values[j] / values[j].sum() for j in range(len(values))]
counts_normalized

fig, axes = plt.subplots(5, 2)
for j in range(len(counts_normalized)):
    ax = sns.lineplot(counts_normalized[j], ax=axes[int(j / 2), j % 2])
    ax.set(xlabel="Label Frequency", ylabel="Clients")
plt.show()

# In[22]:


# gaussian ~ N(5, 0.5)
values = [np.array([count_gaussian_1.get(k).get(key) for key in count_gaussian_1.get(k)]) for k in count_gaussian_1]
counts_normalized = [values[j] / values[j].sum() for j in range(len(values))]
counts_normalized

fig, axes = plt.subplots(5, 2)
for j in range(len(counts_normalized)):
    ax = sns.lineplot(counts_normalized[j], ax=axes[int(j / 2), j % 2])
    ax.set(xlabel="Label Frequency", ylabel="Clients")
plt.show()

# In[23]:


# gaussian ~ N(5, 1000)
values = [np.array([count_gaussian_1000.get(k).get(key) for key in count_gaussian_1000.get(k)]) for k in
          count_gaussian_1]
counts_normalized = [values[j] / values[j].sum() for j in range(len(values))]
counts_normalized

fig, axes = plt.subplots(5, 2)
for j in range(len(counts_normalized)):
    ax = sns.lineplot(counts_normalized[j], ax=axes[int(j / 2), j % 2])
    ax.set(xlabel="Label Frequency", ylabel="Clients")
plt.show()

# In[24]:


# gaussian ~ N(5, 20)
values = [np.array([count_gaussian_20.get(k).get(key) for key in count_gaussian_20.get(k)]) for k in count_gaussian_20]
counts_normalized = [values[j] / values[j].sum() for j in range(len(values))]
counts_normalized

fig, axes = plt.subplots(5, 2)
for j in range(len(counts_normalized)):
    ax = sns.lineplot(counts_normalized[j], ax=axes[int(j / 2), j % 2])
    ax.set(xlabel="Label Frequency", ylabel="Clients")
plt.show()


# In[106]:


# same as sp.stats.entropy(p, q, base=2)
def kl_divergence(p, q):
    return sum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)) if q[i] != 0 and p[i] != 0)


# Jensen-Shannon Divergence
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# In[110]:


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

    mode_entropy = {
        "kl": lambda x: kl_divergence(x, probs_original),
        "js": lambda x: js_divergence(x, probs_original)
    }

    if (mode in mode_entropy):
        try:
            test_func = mode_entropy[mode]
        except KeyError:
            raise ValueError(f"Invalid mode: {mode}")

        stats, pvals = [test_func(probs[j]) for j in range(len(probs))], []
    else:
        try:
            test_func = mode_dict[mode]
        except KeyError:
            raise ValueError(f"Invalid mode: {mode}")

        stats, pvals = zip(*[test_func(probs[j]) for j in range(len(probs))])

    return stats, pvals


# In[111]:


stats_ks_homo, pvals_ks_homo = distance(y_train, map_homo, "kolmogorov-smirnov")
stats_ks_homo, pvals_ks_homo

# In[112]:


stats_dir_05, vals_dir_05 = distance(y_train, map_dir_05, "kolmogorov-smirnov")
stats_dir_05, vals_dir_05

# In[103]:


stats_dir_1000, vals_dir_1000 = distance(y_train, map_dir_1000, "kolmogorov-smirnov")
stats_dir_1000, vals_dir_1000


# In[119]:


def run_experiment(alpha_vector, path, mode_part, mode_test, n_clients, logpath=None, save=False):
    x_train, y_train, x_test, y_test = get_data(path)
    evol_stat, evol_pval = [], []

    for j in range(len(alpha_vector)):
        a_j = alpha_vector[j]
        subset_map = partition(path, mode_part, n_clients, a_j)
        if save:
            tensor_to_csv(x_train, y_train, subset_map, j)
        stats, pvals = distance(y_train, subset_map, mode_test)
        mean_teststat_j = np.mean(stats)
        mean_pval_j = np.mean(pvals)
        evol_stat.append(mean_teststat_j)
        evol_pval.append(mean_pval_j)

    title_formats_part = {
        "hetero-dir": "Mean divergence from original distribution under (increasingly) heterogeneous partitioning via Dirichlet distribution",
        "hetero-gaussian": "Mean divergence from original distribution under (increasingly) heterogeneous partitioning via Gaussian distribution",
        "homo": "A homogeneous partitioning with {n_clients} subsets",
    }

    title_formats_test = {
        "kolmogorov-smirnov": "Test Statistic: Kolmogorov-Smirnov",
        "empirical": "Test Statistic: Empirical Distribution",
        "kl": "(Entropy-Based) Kullback-Leibler Divergence",
        "js": "(Entropy-Based) Jensen-Shannon Divergence"
    }

    def linplot(stats):
        main_title = title_formats_part.get(mode_part, "")
        main_title = main_title.format(n_clients=n_clients)
        appendage = title_formats_test.get(mode_test, "")
        main_title = main_title + '\n' + appendage

        plt.figure(figsize=(5, 3), dpi=300)
        plt.plot(np.arange(len(stats)), stats)
        plt.xlabel([f'α_{j}={alpha}' for j, alpha in enumerate(alpha_vector)])
        plt.xticks(np.arange(len(stats)))
        plt.suptitle(main_title)
        plt.show()

    linplot(evol_stat)


# In[115]:


alpha_vector = [1000, 500, 250, 125, 60, 30, 20, 10, 5, 1, 0.5]
run_experiment(alpha_vector, "D:/", "hetero-dir", "kolmogorov-smirnov", 10)

# In[118]:


alpha_vector = [1000, 500, 250, 125, 60, 30, 20, 10, 5, 1, 0.5]
run_experiment(alpha_vector, "D:/", "hetero-dir", "kl", 10)

# In[ ]:


# In[ ]:
