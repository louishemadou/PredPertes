import keras
import tensorflow as tf
import os
import time
from pca import *
from utils import error
from split import retrieve_and_split, retrieve_all_and_split, retrieve, sample_and_retrieve
from visualize import compare, compare_nd


X_train, Y_train = retrieve(2016)
X_val, Y_val = retrieve(2017)

X_train_n = normalize(X_train)
X_val_n = normalize(X_val)

m = X_train.shape[1]

influe_student = [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
to_select_student = []
not_to_select_student = []
for i in range(len(influe_student)):
    if influe_student[i]:
        to_select_student.append(i)
    else:
        not_to_select_student.append(i)

doublons = [3, 5, 6, 8, 12, 30, 16]
expl = [1, 2, 11, 22, 25]

to_select_doublons = []
to_select_expl = []
to_select_doublon_expl = []
for i in range(m):
    if i not in doublons:
        to_select_doublons.append(i)
    if i not in expl:
        to_select_expl.append(i)
    if i not in doublons:
        if i not in expl:
            to_select_doublon_expl.append(i)


X_train_2n = X_train_n[:, to_select_doublon_expl]
X_val_2n = X_val_n[:, to_select_doublon_expl]
# Testing PCA and creating modified sets



X_train_n = normalize(X_train)
X_val_n = normalize(X_val)


class Net():
    def __init__(self, lr, input_dim, layers, activations):
        self.lr = lr
        self.model = keras.models.Sequential()
        for i in range(len(layers)):
            if i!=0:
                self.model.add(keras.layers.Dense(layers[i], activation=activations[i], input_dim=input_dim))
            else:
                self.model.add(keras.layers.Dense(layers[i], activation=activations[i]))
        print(str(input_dim) + " features")
        self.history = None

    def fit(self, X, Y, epochs = 100, batch_size = 200, Visual = False):
        SGD = keras.optimizers.SGD(learning_rate = self.lr)
        self.model.compile(loss='mean_squared_error', optimizer=SGD, metrics=[keras.metrics.MeanSquaredError()])
        self.history = self.model.fit(X, Y, batch_size, epochs)
        if Visual: 
            plt.plot(self.history.history['loss'])
            plt.title('loss over epoch')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.show()

    def predict(self, X):
        return self.model.predict(X)
"""
T0 = time.time()
layers = [400, 400, 100, 1]
activations = ["sigmoid", "sigmoid", "relu", "linear"]
input_dim = X_train.shape[1]
lr = 0.00001
net = Net(lr, input_dim, layers, activations)
net.fit(X_train_2n, Y_train, epochs = 100, batch_size = 100, Visual = False)
Y_pred = net.predict(X_val_2n)
error(Y_pred, Y_val)
print(time.time() - T0)
#compare(Y_pred, Y_val)
"""
