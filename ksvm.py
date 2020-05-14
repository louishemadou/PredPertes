import numpy as np
import matplotlib.pyplot as plt
from utils import add_ones

def poly_kernel(x1, x2, param):
    return np.dot(x1, x2)**param

def gauss_kernel(x1, x2, param):
    return np.exp(- param * np.linalg.norm(x1 - x2)**2)

class SDCARegressor():
    def __init__(self, kernel, param, C = 10):
        self.kernel = kernel
        self.param = param
        self.C = C
        self.X = None
        self.Y = None
        self.alpha = None
        
    def fit(self, X, Y, epochs = 100, Visual = False):
        
        def dual_Loss(alpha, X, Y, k):
            n = X.shape[0]
            return sum([alpha[i]*Y[i] - (alpha[i]**2)/4 for i in range(n)]) - (1/(2*self.C))*sum([alpha[i]*alpha[j]*k(X[i], X[j], self.param) for i in range(n) for j in range(n)])
        
        X_ones = add_ones(X)
        n, p = X_ones.shape
        self.alpha = np.zeros(n)
        self.Y = Y
        self.X = X_ones
        loss_values = []
        for k in range(epochs):
            perm = np.random.permutation(n)
            for i in perm:
                delta_i = (self.Y[i] - sum([self.alpha[j]*self.kernel(self.X[i], self.X[j], self.param) for j in range(n)]) - (1/2)*self.alpha[i])/((1/2) + self.kernel(self.X[i], self.X[i], self.param)/self.C)
                self.alpha[i] = self.alpha[i] + delta_i
                print('training, iteration: '+str(k+1)+'/'+str(epochs)+'\r', sep=' ', end='', flush=True)
            if Visual:
                loss_values.append(dual_Loss(self.alpha, X_ones, Y, self.kernel))
            
        if Visual:
            it = range(len(loss_values))
            plt.figure()
            plt.plot(it, loss_values, 'r')
            plt.title("Loss over epochs")
            plt.show()
            
    def predict(self, X):
        X_ones = add_ones(X)
        n = self.X.shape[0]
        m = X_ones.shape[0]
        Y_pred = np.array([sum([(1/self.C)*self.alpha[i]*self.kernel(self.X[i], X_ones[j], self.param) for i in range(n)]) for j in range(m)])
        return Y_pred
    
    def weights(self):
        return self.alpha
