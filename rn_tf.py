import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *

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
