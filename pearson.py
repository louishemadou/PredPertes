### test de corrélation (Pearson)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # pour afficher la map de corrélation


# avec scipy (utiliser scipy.stats.pearsonr)

import scipy as sc # H0 = non-correlation (assumption that each dataset is normally distributed)

inner_corr = np.zeros((len(X), len(X)))
target_corr = np.zeros(len(X))
for i in range(len(X)):
    target_corr[i] = sc.stats.pearsonr(X[:, i], Y)[0]
    for j in range(len(X)):
        inner_corr[i, j] = sc.stats.pearsonr(X[:, i], X[:, j])[0]


# avec numpy (utiliser np.corrcoef)


# avec pandas (utiliser pandas.DataFrame.corr)


# afficher la map de corrélation entre features

plt.figure(figsize=(12,10))
sns.heatmap(inner_corr, annot=True)
plt.title("Pearson correlation between features")
plt.show()

# supprimer les "doublons" :
# pour chaque feature, si une feature a une corrélation de plus de tant (voir avec la map), on en supprime une

# garder les features les plus corrélées à Y
# on choisit un nombre de feature ou un seuil de corrélation ?

