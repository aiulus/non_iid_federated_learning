import keras
import pandas as pd
from torchvision import transforms
from keras import datasets, models, layers, optimizers
from keras.datasets import mnist, cifar10
from keras.utils import plot_model
import matplotlib.pyplot as plt
import torch
from torch import from_numpy, tensor

# train, test = datasets.MNIST(root="D:/datasets", train=True, download=True, transform=transforms.ToTensor()), \
#               datasets.MNIST(root="D:/datasets", train=False, download=True, transform=transforms.ToTensor())

train, test = mnist.load_data()


def create_model(alpha, train, test):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(units=256, activation='relu'))
    # First hidden layer
    model.add(layers.Dropout(rate=0.4))
    p = len(set(train.targets.tolist()))
    model.add(layers.Dense(units=p, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(learning_rate=alpha),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_data, train_targets, epochs, batch_size=None, validation_split=0.1):
    history = model.fit(x=train_data, y=train_targets, batch_size=batch_size,
                        epochs=epochs, shuffle=True, validation_split=validation_split)
    epochs=history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

model = create_model(learning_rate, train)
epochs, hist = train_model(model, train.data, train.targets, epochs, batch_size, validation_split)

list_of_metrics_to_plot = ['accuracy']
plt.plot(epochs, hist, list_of_metrics_to_plot)

print("\n Evaluate the new model against the test set:")
model.evaluate(x=test.data, y=test.targets, batch_size=batch_size)