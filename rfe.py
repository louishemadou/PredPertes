from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from scipy.stats import t
from split import retrieve
import numpy as np
import time

X_2016, Y_2016 = retrieve(2016)
X_2017, Y_2017 = retrieve(2017)
X_2018, Y_2018 = retrieve(2018)

n_points = 200
r = np.random.permutation(n_points)
X = np.concatenate((X_2016, X_2017, X_2018))[r[0:n_points],:]
Y = np.concatenate((Y_2016, Y_2017, Y_2018))[r[0:n_points]]
print(X.shape)
estimator = SVR(kernel = "linear")
T0 = time.time()
selector = RFE(estimator, n_features_to_select = 30, verbose = 10)
selector = selector.fit(X, Y)
print(selector.ranking_)
print(time.time()-T0)



labels = ["biais", "perimetre", "nature", "date", "heures", "consommation", "prevision_1", "prevision_0", "fioul", "charbon", "gaz", "nucleaire", "eolien", "solaire", "hydraulique", "pompage", "bioenergies", "echanges_physiques", "taux_co2", "echanges_angleterre", "echanges_espagne", "echanges_italie", "echanges_suisse", "echanges_allemagne_belgique", "fioul_tac", "fioul_cogen", "fioul_autres", "gaz_tac", "gaz_cogen", "gaz_ccg", "gaz_autres", "hydro_ecluses", "hydro_lacs", "hydro_turbines" "bio_dechets", "bio_biomasse", "bio_biogaz"]

