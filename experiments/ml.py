import keras
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)