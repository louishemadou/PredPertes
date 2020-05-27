import keras
import tensorflow as tf
import os
from pca import *
from utils import error
from split import retrieve_and_split, retrieve_all_and_split, retrieve
from visualize import compare

os.environ['TF_KERAS'] = '1'

X_train, Y_train = retrieve(2016)
X_val, Y_val = retrieve(2017)

# Testing PCA and creating modified sets

m = X_train.shape[1]

X_train_n = normalize(X_train)
X_val_n = normalize(X_val)

X_train_s = standardize(X_train)
X_val_s = standardize(X_val)

X_train_pca = pca(X_train_s, m, whitening = True, visual = False)
X_val_pca = pca(X_val_s, m, whitening = True, visual = False)


class Net():
    def __init__(self, lr, input_dim, layers, activations):
        self.lr = lr
        self.model = keras.models.Sequential()
        for i in range(len(layers)):
            if i!=0:
                self.model.add(keras.layers.Dense(layers[i], activation=activations[i], input_dim=input_dim))
            else:
                self.model.add(keras.layers.Dense(layers[i], activation=activations[i]))
        self.history = None

    def fit(self, X, Y, epochs = 50, batch_size = 100, Visual = False):
        SGD = keras.optimizers.SGD(learning_rate = self.lr)
        self.model.compile(loss='mean_squared_error', optimizer=SGD, metrics=[keras.metrics.MeanSquaredError()])
        print(epochs)
        self.history = self.model.fit(X, Y, batch_size, epochs)
        if Visual: 
            plt.plot(self.history.history['loss'])
            plt.title('loss over epoch')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.show()

    def predict(self, X):
        return self.model.predict(X)

layers = [200, 200, 50, 1]
activations = ["sigmoid", "sigmoid", "relu", "linear"]
input_dim = X_train_s.shape[0]
lr = 0.00001
net = Net(lr, input_dim, layers, activations)
net.fit(X_train_n, Y_train, epochs = 50, batch_size = 100, Visual = True)
Y_pred = net.predict(X_val_n)
error(Y_pred, Y_val)
compare(Y_pred, Y_val)
