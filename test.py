from sklearn.metrics import r2_score
from sklearn.svm import SVR
from split import retrieve_and_split, retrieve_all_and_split
from pca import standardize, normalize, pca
from visualize import compare
from linear_r import LinearRegression
from utils import error
from svm import SAGRegressor
from ksvm import poly_kernel, gauss_kernel, SDCARegressor
from rn import LeastSquareCriterion, MLP

X_train, Y_train, X_val, Y_val, X_test, Y_test = retrieve_all_and_split()
m = X_train.shape[1]
print(str(m) + " features")

# Testing PCA and creating modified sets

X_train_n = normalize(X_train)
X_val_n = normalize(X_val)

X_train_s = standardize(X_train)
X_val_s = standardize(X_val)

X_train_pca = pca(X_train_s, m, whitening = True, visual = False)
X_val_pca = pca(X_val_s, m, whitening = True, visual = False)

# Testing Linear_regression
"""""
# With normalized data

print("Linear regression with normalized data")
LR = LinearRegression(lamb = 0.001, delta = 0.00001)
LR.fit(X_train_n, Y_train, epochs = 200, Visual = True)
Y_pred = LR.predict(X_val_n)
error(Y_pred, Y_val)
compare(Y_val, Y_pred)

# With standardized data

print("Linear regression with standardized data")
LR = LinearRegression(lamb = 0.001, delta = 0.00001)
LR.fit(X_train_s, Y_train, epochs = 200, Visual = True)
Y_pred = LR.predict(X_val_s)
error(Y_pred, Y_val)
compare(Y_val, Y_pred)


# With standardized and orthogonalized data

print("Linear regression with standardized and orthogonalized data")
LR = LinearRegression(lamb = 0, delta = 0.0001)
LR.fit(X_train_pca, Y_train, epochs = 200, Visual = True)
Y_pred = LR.predict(X_val_pca)
error(Y_pred, Y_val)
compare(Y_val, Y_pred)
"""
# Testing SVR

# With normalized data

print("SVR with normalized data")
svr = SVR(kernel = 'rbf')
svr.fit(X_train_n, Y_train)
Y_pred = svr.predict(X_val_n)
error(Y_pred, Y_val)
compare(Y_val, Y_pred)
# With standardized data

print("SVR with standardized data")
svr = SVR(kernel = 'rbf')
svr.fit(X_train_s, Y_train)
Y_pred = svr.predict(X_val_s)
error(Y_pred, Y_val)

# With standardized and orthogonalized data

print("SVR with standardized and orthogonalized data")
svr = SVR(kernel = 'rbf')
svr.fit(X_train_pca, Y_train)
Y_pred = svr.predict(X_val_pca)
error(Y_pred, Y_val)

