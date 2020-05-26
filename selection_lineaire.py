from linear_r import LinearRegression
from split import retrieve
from visualize import influence
from pca import standardize
from utils import add_ones
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
LR.fit(X, Y, epochs=15, Visual=True)
Y_pred = LR.predict(X)

w = LR.weights()

labels = ["biais", "perimetre", "nature", "date", "heures", "consommation", "prevision_1", "prevision_0", "fioul", "charbon", "gaz", "nucleaire", "eolien", "solaire", "hydraulique", "pompage", "bioenergies", "echanges_physiques", "taux_co2", "echanges_angleterre", "echanges_espagne", "echanges_italie", "echanges_suisse", "echanges_allemagne_belgique", "fioul_tac", "fioul_cogen", "fioul_autres", "gaz_tac", "gaz_cogen", "gaz_ccg", "gaz_autres", "hydro_ecluses", "hydro_lacs", "hydro_turbines" "bio_dechets", "bio_biomasse", "bio_biogaz"]

X_ones = add_ones(X)
inv = np.linalg.inv(X_ones.T @ X_ones)
beta_hat = (inv @ X_ones.T) @ Y
sigma_hat = np.linalg.norm(Y_pred - Y)/np.sqrt(n-p-1)
alpha = 0.05
f = t.ppf(1-alpha/2, n-p-1)

output = []

for i in range(p+1): # Formule du poly
    rho = inv[i, i]
    beta = beta_hat[i]
    output.append([labels[i], w[i], abs(beta/np.sqrt(rho*sigma_hat**2)) >= f])

Influence = [out[0] for out in output if out[2]]
NInflience = [out[0] for out in output if not out[2]]

print("Variables ayant une influence: ", Influence)
print("Variables n'ayant pas d'influence ", NInflience)



"""
plt.figure()
plt.xticks(range(p+1), labels, rotation = "vertical")
plt.margins(0.05)
plt.subplots_adjust(bottom = 0.30)
fig, ax = plt.subplots()
N, bins, patches = ax.hist(w, edgecolor='white', linewidth=1)

plt.show()
"""
