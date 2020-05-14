"""Useful functions for algorithms"""

import numpy as np
from sklearn.metrics import r2_score

def accuracy(Y_pred, Y):
    """Measures accuracy of a
    regression prediction"""
    error = np.abs(Y_pred - Y)
    exp_error = 100*np.mean(error)/np.mean(Y)
    r2 = r2_score(Y, Y_pred)
    print("R² score: " + str(r2))
    print(str(round(exp_error, 2)) + " % experimental error") # Mean experimental error

def add_ones(X):
    """ Adds a column of ones in the beginning of X"""
    new_X = np.array([np.ones(X.shape[0])] + [sample for sample in X.T])
    return new_X.T
