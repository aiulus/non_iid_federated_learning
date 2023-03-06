import tensorflow as tf
from torchvision.datasets import MNIST
import numpy as np

train = MNIST(root="D:/", train=True, download=False, transform=None)
test = MNIST(root="D:/", train=False, download=False, transform=None)

# defining the network
input_width = 28
input_height = 28
input_channels = 1
input_pixels = 784

n_conv1 = 32
n_conv2 = 64
stride_conv1 = 1
stride_conv2 = 1
conv1_k = 5
conv2_k = 5
max_pool1_k = 2
max_pool2_k = 2

n_hidden = 1024
n_out = 10

input_size_to_hidden = (input_width//(max_pool1_k*max_pool2_k)) * (input_height//(max_pool1_k*max_pool2_k)) * n_conv2

# Initialising the weights with random values
weights = {
    "wc1" : tf.Variable(tf.random_normal_initializer([conv1_k, conv1_k, input_channels, n_conv1])),
    "wc2" : tf.Variable(tf.random_normal_initializer([conv2_k, conv2_k, n_conv1, n_conv2])),
    "wh1": tf.Variable(tf.random_normal_initializer([input_size_to_hidden, n_hidden])),
    "wo" : tf.Variable(tf.random_normal_initializer([n_hidden, n_out]))
}

biases = {
    "bc1" : tf.Variable(tf.random_normal_initializer([n_conv1])),
    "bc2" : tf.Variable(tf.random_normal_initializer([n_conv2])),
    "bh1" : tf.Variable(tf.random_normal_initializer([n_hidden])),
    "bo" : tf.Variable(tf.random_normal_initializer([n_out]))
}

# Functions for layers
def conv(x, weights, bias, strides=1):
    out = tf.nn.conv2d(x, weights, padding="SAME", strides = [1, strides, strides, 1])
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)
    return out

def maxpooling(x, k = 2):
    return tf.nn.max_pool(x, padding="SAME", ksize=[1, k, k, 1], strides=[1, k, k, 1])

# Forward pass
def cnn(x, weights, biases, keep_prob):
    x = tf.reshape(x, shape = [-1, input_height, input_width, input_channels])
    conv1 = conv(x, weights['wc1'], biases['bc1'], stride_conv1)
    conv1_pool = maxpooling(conv1, max_pool1_k)

    conv2 = conv(conv1_pool, weights['wc2'], biases['bc2'], stride_conv2)
    conv2_pool = maxpooling(conv2, max_pool2_k)

    hidden_input = tf.reshape(conv2_pool, shape = [-1, input_size_to_hidden])
    hidden_output_before_activation = tf.add(tf.matmul(hidden_input, weights['wh1']), biases['bh1'])
    hidden_output_before_activation_X = tf.nn.relu(hidden_output_before_activation)
    hidden_output = tf.nn.dropout(hidden_output_before_activation_X, keep_prob)

    output = tf.add(tf.matmul(hidden_output, weights['wo']), biases['bo'])
    return output

##################################################################
from sklearn import preprocessing
from keras import models
from keras import layers

# Preprocessing

features = np.array([[-100.1, 3240.1],
                     [-200.2, -234.1],
                     [5000.5, 150.1],
                     [6000.6, -125.1],
                     [9000.9, -673.1]])
# eq. to:
# def scaler(a):
#     for i in range(a.shape[1]):
#         mean_i = a[:, i].mean()
#         std_i = a[:, i].std()
#         a[:, i] = (a[:, i] - mean_i)/std_i
#     return a

scaler = preprocessing.StandardScaler()

features_standardized = scaler.fit_transform(features)

# Start NN
network = models.Sequential()

# Add fully connected layer with a ReLU acivation function // units: #nodes // In keras, the first hidden layer of any
# network has to include an input_shape parameter, which is the shape of the input data
network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu"))

# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation="sigmoid"))

# Compile NN
network.compile(loss="binary_crossentropy", # Cross-entropy
                optimizer="rmsprop", # Root Mean Square Propagation
                metrics=["accuracy"]) # Accuracy performance metric

###############################################
# Saving and Loading Trained Models

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import joblib

iris = datasets.load_iris()
features = iris.data
target = iris.target

classifier = RandomForestClassifier()

# Train model
model = classifier.fit(features, target)

# Save model as a pickle file
joblib.dump(model, "model_iris_RF.pkl")

# Load model from file
classifier = joblib.load("model_iris_RF.pkl")

new_observation = [[5.2, 3.2, 1.1, 0.1]]
classifier.predict(new_observation)

# Saved models might not be compatible between versions of scikit-learn; therefore, it can be helpful to
# include the version of scikit-learn used in the model in the filename:
import sklearn

scikit_version = sklearn.__version__

joblib.dump(model, "model_{version}.pkl".format(version = scikit_version))

################################################################################
# Saving and Loading a Keras Model

# Save the model as HDF5:
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.models import load_model

n_features = 1000

(train_X, train_Y), (test_X, test_Y) = imdb.load_data(num_words=n_features)

# Convert movie review data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=n_features)
train_features = tokenizer.sequences_to_matrix(train_X, mode="binary")
test_features = tokenizer.sequences_to_matrix(test_X, mode="binary")

network = models.Sequential()

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16,
                         activation="relu",
                         input_shape=(n_features,)))

# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation="sigmoid"))

# Compile neural network
network.compile(loss="binary_crossentropy", # Cross-entropy
                optimizer="rmsprop", # Root Mean Square Propagation
                metrics=["accuracy"]) # Accuracy performance metric

# Train NN
history = network.fit(train_features,
                      train_Y,
                      epochs=3,
                      verbose=0,
                      batch_size=100,
                      validation_data=(test_features, test_Y))

network.save("model_imdb_keras.h5")

# network = load_model("model_imdb_keras.h5")

#######################################################################
# Model Evaluation

# Cross-Validation
from sklearn import datasets, metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits()
features, target = digits.data, digits.target

standardizer = StandardScaler()
logit = LogisticRegression()

# Create a pipeline that standardizes, then runs logistic regression
pipeline = make_pipeline(standardizer, logit)

# Create k-Fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=1)
cv_results = cross_val_score(pipeline,
                             features,
                             target,
                             cv=cv,
                             scoring="accuracy",
                             n_jobs=-1) # Use all CPU cores
cv_results.mean()

# Evaluating Binary Classifier Predictions
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate features matrix and target vector
X, y = make_classification(n_samples=10000,
                           n_features=3,
                           n_informative=3,
                           n_redundant=0,
                           n_classes=2,
                           random_state=1)
logit = LogisticRegression()

cross_val_score(logit, X, y, scoring="accuracy")

# Evaluating Multiclass Classifier Predictions

# Use metrics that can handle multiple classes
cross_val_score(logit, X, y, scoring="accuracy")
cross_val_score(logit, X, y, scoring="f1_macro")
# macro: Calculate mean of metric scores for each class
# weighted: Weighted average of metric scores for each class
# micro: Mean of metric scores for each obs.-class combination

# Visualizing a Classifier's Performance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

iris = datasets.load_iris()
features, targets = iris.data, iris.target
class_names = iris.target_names

features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)
classifier = LogisticRegression()
target_predicted = classifier.fit(features_train, target_train).predict(features_test)
matrix = confusion_matrix(target_test, target_predicted)
dataframe = pd.DataFrame(matrix, index = class_names, columns = class_names)

sns.heatmap(dataframe, annot= True, cbar=None, cmap='Blues')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

# Visualizing the Effects of Training Set Size
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

digits = load_digits()
features, target = digits.data, digits.target

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(),
                                                        features,
                                                        target,
                                                        cv=10,
                                                        scoring='accuracy',
                                                        n_jobs=-1,
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, 1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")


plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()

################### MODEL SELECTION ########################

# 1. Exhaustive Search
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
features = iris.data
target = iris.target

logistic = linear_model.LogisticRegression()
# Create a range of candidate penalty hyperparameter values
penalty = ['11', '12']
# Create a range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
best_model = gridsearch.fit(features, target)

# 2. Randomized Search
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# Load data
iris = datasets.load_iris()
features, targets = iris.data, iris.targets

logistic = linear_model.LogisticRegression()
penalty = ['11', '12']
C = uniform(loc=0, scale=4)
hyperparameters = dict(C=C, penalty=penalty)

randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

best_model = randomizedsearch.fit(features, targets)


def run_experiment(dataset_type, alpha_vector, path, mode_part, mode_test, n_clients, logpath=None, save=False):
    train, test = get_data(dataset_type, path)
    x_train, y_train, x_test, y_test = train.data, train.targets, test.data, test.targets
    evol_stat, evol_pval = [], []

    for j in range(len(alpha_vector)):
        a_j = alpha_vector[j]
        subset_map = partition(train, mode_part, n_clients, a_j)
        if save:
            tensor_to_csv(x_train, y_train, subset_map, j)
        stats, pvals = distance(y_train, subset_map, mode_test)
        mean_teststat_j = np.mean(stats)
        mean_pval_j = np.mean(pvals)
        evol_stat.append(mean_teststat_j)
        evol_pval.append(mean_pval_j)

    title_formats_part = {
        "hetero-dir": "Mean divergence from original distribution under heterogeneous partitioning via Dirichlet distribution",
        "hetero-gaussian": "Mean divergence from original distribution under heterogeneous partitioning via Gaussian distribution",
        "homo": "A homogeneous partitioning with {n_clients} subsets",
        "quant": "A quantity-based heterogeneous partitioning with {n_clients} subsets"
    }

    title_formats_test = {
        "kolmogorov-smirnov": "Test Statistic: Kolmogorov-Smirnov",
        "empirical": "Test Statistic: Empirical Distribution",
        "kl": "Test Statistic: Kullback-Leibler Divergence (Entropy-Based)",
        "js": "Test Statistic: Jensen-Shannon Divergence (Entropy-Based)",
        "wd": "Test Statistic: Wasserstein Distance",
        "gini": "Test Statistic: Gini Coefficient"
    }

    def linplot(stats):
        main_title = title_formats_part.get(mode_part, "")
        main_title = main_title.format(n_clients=n_clients)
        appendage = title_formats_test.get(mode_test, "")
        main_title = dataset_type + ': ' + main_title + '\n' + appendage

        plt.figure(figsize=(5, 3), dpi=300)
        plt.plot(np.arange(len(stats)), stats)

        if mode_part == "quant":
            l, u = tuple(round(x, 1) for x in (min(alpha_vector), max(alpha_vector)))
            label = f'[{u}:{l}], set of divisors for {u}'
            plt.xlabel(label)
        elif len(alpha_vector) <= 15:
            plt.xlabel([f'α_{j}={alpha}' for j, alpha in enumerate(alpha_vector)])
            plt.xticks(np.arange(len(stats)))
        else:
            l, u, steps = tuple(round(x, 1) for x in (
            min(alpha_vector), max(alpha_vector), (max(alpha_vector) - min(alpha_vector)) / (len(alpha_vector) - 1)))
            label = f'[{l}:{u}], step size = {steps}'
            plt.xlabel(label)
            # plt.xticks(np.arange(l-steps, u+steps, u/10))
        plt.suptitle(main_title, fontsize='small')
        plt.show()

    linplot(evol_stat)






















