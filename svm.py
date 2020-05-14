import numpy as np
import matplotlib.pyplot as plt
from utils import add_ones

class SAGRegressor():
    def __init__(self, lamb = 0.001, delta = 0.01):
        self.lamb = lamb # Regularization
        self.delta = delta # Learning rate
        self.w = None # Weights and biais
    
    def fit(self, X, Y, epochs = 100, Visual = False):

        def Loss(w, lamb, X, Y):
            n = X.shape[0]
            return (1/n) * sum(((Y[i] - np.dot(w, X[i]))**2 for i in range(n))) + lamb * np.linalg.norm(w)**2

        def grad_f(w, x_i, y_i):
            return -2*x_i*(y_i - np.dot(w, x_i))
            
        X_ones = add_ones(X)
        n, p = X_ones.shape
        d = np.zeros(p)
        z = np.zeros((n, p)) # Remembering gradients
        self.w = np.zeros(p)
        visit = np.zeros(n) # Visited samples
        loss_values = []
        for k in range(epochs):
            i = np.random.randint(0, n)
            visit[i] = 1
            d = d - z[i] + grad_f(self.w, X_ones[i], Y[i])
            z[i] = grad_f(self.w, X_ones[i], Y[i])
            m = np.sum(visit)
            reg = self.w
            reg[0] = 0
            self.w = self.w - self.lamb*self.delta*reg - (self.delta/m) * d
            loss_values.append(np.linalg.norm(Loss(self.w, self.lamb, X_ones, Y)))
            print('training, iteration: '+str(k+1)+'/'+str(epochs)+'\r', sep=' ', end='', flush=True)
            
        if Visual:
            it = range(len(loss_values))
            plt.figure()
            plt.plot(it, loss_values, 'r')
            plt.title("Loss over epochs")
            plt.show()
            
    def predict(self, X):
        X_ones = add_ones(X)
        Y_pred = np.dot(X_ones, self.w)
        return Y_pred
    
    def weights(self):
        return self.w
