"""Useful functions for algorithms"""

import numpy as np
import os
from sklearn.metrics import r2_score

def error(Y_pred, Y, set_name):
    """Measures accuracy of a
    regression prediction"""
    r2 = r2_score(Y, Y_pred)
    print("R² score for "+ set_name + " = " + str(r2))
    n = len(Y_pred)
    mape = (1/n)*sum(abs((Y_pred[i]-Y[i])/Y[i]) for i in range(n))
    print("MAPE score for " + set_name + " = " + str(mape))
    rmse = np.sqrt((1/n)*sum((Y_pred[i]-Y[i])**2 for i in range(n)))
    print("RMSE score for  "+ set_name + " = "+ str(rmse))

def add_ones(X):
    """ Adds a column of ones in the beginning of X"""
    ones = np.ones(X.shape[0])
    ones.shape = (X.shape[0], 1)
    return np.concatenate((ones, X), axis=1)

def path(name):
    if os.name != 'posix':
        return name.replace('/', '\\')
    return name
