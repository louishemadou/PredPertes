import numpy as np
import pandas as pd


def retrieve(year):
    """Retrieve data"""
    path_X = "./data/conso/conso_" + str(year) + ".csv"
    path_Y = "./data/pertes/pertes_" + str(year) + ".csv"
    X = pd.read_csv(path_X).to_numpy()
    Y = pd.read_csv(path_Y).to_numpy()[:, 3]

    return X, Y

def retrieve_features():
    """Return all features name"""
    return pd.read_csv("./data/conso/conso_2015.csv").columns

def retrieve_and_split(year):
    """Retrieve and split data
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

