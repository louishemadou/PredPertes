from linear_r import LinearRegression
from visualize import influence
from utils import *
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt

X_2016, Y_2016 = retrieve(2016)
X_2017, Y_2017 = retrieve(2017)
X_2018, Y_2018 = retrieve(2018)

X = np.concatenate((X_2016, X_2017, X_2018))
Y = np.concatenate((Y_2016, Y_2017, Y_2018))

n, p = X.shape
LR = LinearRegression(lamb=1000, delta=0.00000000000000005)
LR.fit(X, Y, epochs=100, Visual=True)
Y_pred = LR.predict(X)

w = LR.weights()

labels = ["biais"] + list(retrieve_features())
X_ones = add_ones(X)
inv = np.linalg.inv(X_ones.T @ X_ones)
beta_hat = (inv @ X_ones.T) @ Y
Y_hat = X_ones @ beta_hat
sigma_hat = np.linalg.norm(Y_hat - Y)/np.sqrt(n-p-1)
alpha = 0.05
f = t.ppf(1-alpha/2, n-p-1)

output = []

for i in range(p+1): # Formule du poly
    rho = inv[i, i]
    beta = beta_hat[i]
    output.append([labels[i], w[i], abs(beta/np.sqrt(rho*sigma_hat**2)) >= f])

influe = []

for i in range(p+1): # Tests de Student
    print(w[i], labels[i])
    if i>0:
        influe.append(int(output[i][2]))

Influence = [out[0] for out in output if out[2]]
NInflience = [out[0] for out in output if not out[2]]
print("Variables ayant une influence: ", Influence)
print(len(Influence))
print("Variables n'ayant pas d'influence ", NInflience)
print(len(Influence))


