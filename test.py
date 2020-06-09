from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from split import retrieve 
from pca import standardize, normalize
from visualize import compare
from linear_r import LinearRegression
from utils import error
from svm import SAGRegressor
from ksvm import poly_kernel, gauss_kernel, SDCARegressor
from rn_tf import *

X_train, Y_train = retrieve(2016)
X_val, Y_val = retrieve(2017)
X_test, Y_test = retrieve(2018)
m = X_train.shape[1]
print(str(m) + " features")

# Testing PCA and creating modified sets

X_train_n = normalize(X_train)
X_val_n = normalize(X_val)
X_test_n = normalize(X_test)

X_train_s = standardize(X_train)
X_val_s = standardize(X_val)
X_test_s = standardize(X_test)

pca = PCA(n_components=m, whiten = True)

X_train_pca = pca.fit_transform(X_train_s)
X_val_pca = pca.fit_transform(X_val_s)
X_test_pca = pca.fit_transform(X_test_s)

# Testing Linear_regression
# With normalized data

print("Linear regression with normalized data")
LR = LinearRegression(lamb = 0.1, delta = 0.000015)
LR.fit(X_train_n, Y_train, epochs = 1000, Visual = True)
Y_pred_1 = LR.predict(X_val_n)
Y_pred_2 = LR.predict(X_test_n)
error(Y_pred_1, Y_val, "val")
error(Y_pred_2, Y_test, "test")
#compare(Y_val, Y_pred)
# With standardized data

print("Linear regression with standardized data")
LR = LinearRegression(lamb = 0.1, delta = 0.00001)
LR.fit(X_train_s, Y_train, epochs = 1000, Visual = True)
Y_pred_1 = LR.predict(X_val_s)
Y_pred_2 = LR.predict(X_test_s)
error(Y_pred_1, Y_val, "val")
error(Y_pred_2, Y_test, "test")
compare(Y_test, Y_pred_2, 2018)

# With standardized and orthogonalized data

print("Linear regression with standardized and orthogonalized data")
LR = LinearRegression(lamb = 0.1, delta = 0.00001)
LR.fit(X_train_pca, Y_train, epochs = 200, Visual = True)
Y_pred_1 = LR.predict(X_val_pca)
Y_pred_2 = LR.predict(X_test_pca)
error(Y_pred_1, Y_val, "val")
error(Y_pred_2, Y_test, "test")
compare(Y_test, Y_pred_2, 2018)

# Testing SVR

# With normalized data
ker = "linear"
print("SVR with normalized data")
svr = SVR(kernel = ker, tol = 1e-3, verbose = True)
svr.fit(X_train_n, Y_train)
Y_pred_1 = svr.predict(X_val_n)
Y_pred_2 = svr.predict(X_test_n)
error(Y_pred_1, Y_val, "val")
error(Y_pred_2, Y_test, "test")

# With standardized data

print("SVR with standardized data")
svr = SVR(kernel = ker, tol = 1e-10, verbose = True)
svr.fit(X_train_s, Y_train)
Y_pred_1 = svr.predict(X_val_s)
Y_pred_2 = svr.predict(X_test_s)
error(Y_pred_1, Y_val, "val")
error(Y_pred_2, Y_test, "test")

# With standardized and orthogonalized data

print("SVR with standardized and orthogonalized data")
svr = SVR(kernel = ker, tol = 1e-10, verbose = True)
svr.fit(X_train_pca, Y_train)
Y_pred_1 = svr.predict(X_val_pca)
Y_pred_2 = svr.predict(X_test_pca)
error(Y_pred_1, Y_val, "val")
error(Y_pred_2, Y_test, "test")

# Testing NN

layers = [400, 400, 100, 1]
activations = ["sigmoid", "sigmoid", "relu", "linear"]
input_dim = X_train.shape[1]
lr = 0.00002
net = Net(lr, input_dim, layers, activations)
net.fit(X_train_s, Y_train, epochs = 50, batch_size = 50, Visual = False)
Y_pred_1 = net.predict(X_val_s)
Y_pred_2 = net.predict(X_test_s)
error(Y_pred_1, Y_val, "val")
error(Y_pred_2, Y_test, "test")
compare(Y_test, Y_pred_2, 2018)

