import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import pandas as pd

# Retrieving and shuffling data

X = pd.read_csv("./data/conso/conso_2016.csv").to_numpy()
Y = pd.read_csv("./data/pertes/pertes_2016.csv").to_numpy()[:, -1]
r = np.random.permutation(len(X))

X_train = X[r[0:6000], :]
Y_train = Y[r[0:6000]]
X_val = X[r[6000:7500], :]
Y_val = Y[r[6000:7500]]
X_test = X[r[7500:], :]
Y_test = Y[r[7500:]]

# Data preprocessing

def standardize(X):
    eps = 1e-8 # In case where std = 0
    X_s = (X-np.mean(X, axis = 0))/(np.std(X, axis = 0) + eps)
    return X_s

def normalize(X):
    eps = 1e-8 # In case where max = min
    X_n = (X-np.min(X, axis=0))/(np.max(X, axis = 0)-np.min(X, axis=0)+eps)
    return X_n

X_t = standardize(X_train)
print(str(X_t.shape[1]) + " features")

def pca(X, n_comp, whitening = True, visual = False):

    K = np.cov(X.T) # Correlation matrix
    eig_values, eig_vectors = eigh(K) # Sorted eigenvalues and eigenvectors

    eig_values = eig_values[::-1]
    eig_vectors = eig_vectors.T[::-1]

    def data_saved(m): # Which percentage of data is saved if we select the m greatest eigenvalues ?
        return sum(eig_values[:m])/sum(eig_values)

    if visual:
        plt.figure(1)
        plt.xlabel("Number of principal component selected")
        plt.ylabel("Percentage of information conserved")
        n, p = X.shape
        x_axis = range(0, p)
        y_axis = [100*data_saved(x) for x in x_axis]
        plt.plot(x_axis,y_axis)
        plt.show()
        print(str(100*data_saved(n_comp)) + "% of data conserved")

    # Orthogonalizing data

    new_base = eig_vectors[:n_comp]
    X_pca = np.dot(X, new_base.T)

    if whitening:
        X_pca = np.dot(X_pca, np.diag(eig_values[0:n_comp]**(-1/2)))

    return X_pca

X_t = standardize(X_train)
n_comp = 20
X_t_pca = pca(X_t, n_comp, visual = True)

