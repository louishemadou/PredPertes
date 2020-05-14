import numpy as np
import matplotlib.pyplot as plt
from utils import add_ones

class LinearRegression():
    def __init__(self, lamb = 0.001, delta = 0.01):
        self.lamb = lamb # Regularization (Ridge)
        self.delta = delta # Learning rate
        self.beta = None # Weights and biais
    
    def fit(self, X, Y, epochs = 100, Visual = False):

        def Loss(X, Y, lamb, beta):
            n = X.shape[0]
            return (1/n) * sum((np.dot(X[i], beta) - Y[i])**2 for i in range(n)) + lamb * np.linalg.norm(beta)**2

        def grad_loss(X, Y, lamb, beta):
            """Computes gradient of the loss
            function"""
            n = X.shape[0]
            Y_pred = np.dot(X, beta)
            dbeta = 2 * (np.sum([(Y_pred[i]-Y[i]) * X[i] for i in range(n)], axis=0)) + 2*lamb*beta
            return dbeta
            
        X_ones = add_ones(X)
        n, p = X_ones.shape
        loss_values = []
        self.beta = np.zeros(p)
    
        for i in range(epochs): # Full Gradient algorithm
            dbeta = grad_loss(X_ones, Y, self.lamb, self.beta)
            self.beta = self.beta - self.delta * dbeta
            print('training, iteration: '+str(i+1)+'/'+str(epochs)+'\r', sep=' ', end='', flush=True)
            if Visual:
                loss_values.append(Loss(X_ones, Y, self.lamb, self.beta))

        if Visual:
            it = range(len(loss_values))
            plt.figure()
            plt.plot(it, loss_values, 'r')
            plt.title("Loss over epochs")
            plt.show()
            
    def predict(self, X):
        X_ones = add_ones(X)
        Y_pred = np.dot(X_ones, self.beta)
        return Y_pred
    
    def weights(self):
        return self.beta

