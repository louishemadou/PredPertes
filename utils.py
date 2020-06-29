"""Useful functions for algorithms"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score

def standardize(X):
    eps = 1e-8 # In case where std = 0
    X_s = (X-np.mean(X, axis = 0))/(np.std(X, axis = 0) + eps)
    return X_s

def normalize(X):
    eps = 1e-8 # In case where max = min
    X_n = (X-np.min(X, axis=0))/(np.max(X, axis = 0)-np.min(X, axis=0)+eps)
    return X_n

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


def retrieve(year):
    """Retrieve data of a single year"""
    path_X = "./data/conso/conso_" + str(year) + ".csv"
    path_Y = "./data/pertes/pertes_" + str(year) + ".csv"
    X = pd.read_csv(path_X).to_numpy()
    Y = pd.read_csv(path_Y).to_numpy()[:, 3]

    return X, Y

def retrieve_features():
    """Return all features name"""
    return pd.read_csv("./data/conso/conso_2015.csv").columns

def retrieve_and_split(year):
    """Retrieve and split data of a single year
    into train/val/test datasets"""
    size_1 = 0.70
    size_2 = 0.85

    path_X = "./data/conso/conso_" + str(year) + ".csv"
    path_Y = "./data/pertes/pertes_" + str(year) + ".csv"
    X = pd.read_csv(path_X).to_numpy()
    Y = pd.read_csv(path_Y).to_numpy()[:, 3]

    n = len(X)
    n_1 = int(size_1*n)
    n_2 = int(size_2*n)

    r = np.random.permutation(n)

    X_train = X[r[0:n_1], :]
    Y_train = Y[r[0:n_1]]
    X_val = X[r[n_1:n_2], :]
    Y_val = Y[r[n_1:n_2]]
    X_test = X[r[n_2:], :]
    Y_test = Y[r[n_2:]]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def retrieve_all_and_split():
    """Retrieve and split data 
    of 2016-2017-2018
    into train/val/test datasets"""

    path_train_X = "./data/conso/conso_2016.csv"
    path_train_Y = "./data/pertes/pertes_2016.csv"
    path_val_X = "./data/conso/conso_2017.csv"
    path_val_Y = "./data/pertes/pertes_2017.csv"
    path_test_X = "./data/conso/conso_2018.csv"
    path_test_Y = "./data/pertes/pertes_2018.csv"
    
    X_train = pd.read_csv(path_train_X).to_numpy()
    Y_train = pd.read_csv(path_train_Y).to_numpy()[:, 3]
    X_val = pd.read_csv(path_val_X).to_numpy()
    Y_val = pd.read_csv(path_val_Y).to_numpy()[:, 3]
    X_test = pd.read_csv(path_test_X).to_numpy()
    Y_test = pd.read_csv(path_test_Y).to_numpy()[:, 3]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def sample_and_retrieve(n):
    """returns n random samples
    of all data"""
    X_2016, Y_2016 = retrieve(2016)
    X_2017, Y_2017 = retrieve(2017)
    X_2018, Y_2018 = retrieve(2018)

    X = np.concatenate((X_2016, X_2017, X_2018))
    Y = np.concatenate((Y_2016, Y_2017, Y_2018))
    r = np.random.permutation(len(Y))
    return X[r[0:n],:], Y[r[0:n]]

