from sklearn.metrics import r2_score
from split import retrieve_and_split, retrieve_all_and_split
from pca import standardize, normalize, pca
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

X_train_pca = pca(X_train_s, m, whitening = True, visual = True)
X_val_pca = pca(X_val_s, m, whitening = True, visual = False)

# Testing Linear_regression

# With normalized data

print("Linear regression with normalized data")
LR = LinearRegression(lamb = 0.001, delta = 0.00001)
LR.fit(X_train_n, Y_train, epochs = 20, Visual = True)
Y_pred = LR.predict(X_val_n)
error(Y_pred, Y_val)


# With standardized data

print("Linear regression with standardized data")
LR = LinearRegression(lamb = 0.001, delta = 0.00001)
LR.fit(X_train_s, Y_train, epochs = 20, Visual = True)
Y_pred = LR.predict(X_val_s)
error(Y_pred, Y_val)


# With standardized and orthogonalized data

print("Linear regression with standardized and orthogonalized data")
LR = LinearRegression(lamb = 0, delta = 0.0001)
LR.fit(X_train_pca, Y_train, epochs = 20, Visual = True)
Y_pred = LR.predict(X_val_pca)
error(Y_pred, Y_val)

# Testing SVM

# With normalized data

print("SVM regression with normalized data")
sag = SAGRegressor(lamb = 0.15, delta = 0.10)
sag.fit(X_train_n, Y_train, epochs = 200, Visual = True)
Y_pred = sag.predict(X_val_n)
error(Y_pred, Y_val)

# With standardized data

print("SVM regression with standardized data")
sag = SAGRegressor(lamb = 50, delta = 0.01)
sag.fit(X_train_s, Y_train, epochs = 100, Visual = True)
Y_pred = sag.predict(X_val_s)
error(Y_pred, Y_val)

# With standardized and orthogonalized data
print("SVM regression with standardized and orthogonalized data")
sag = SAGRegressor(lamb = 10, delta = 0.08)
sag.fit(X_train_pca, Y_train, epochs = 100, Visual = True)
Y_pred = sag.predict(X_val_pca)
error(Y_pred, Y_val)

# Testing Kernel SVM

n_points = 200  # Reducing size of training set to avoid long processing time

X_train_n_2 = X_train_n[0:n_points]

X_train_s_2 = X_train_s[0:n_points]

X_train_pca_2 = X_train_pca[0:n_points]


# With normalized data
print("Kernel SVM regression with normalized data")
sdca = SDCARegressor(gauss_kernel, param = 0.001, C = 1)
sdca.fit(X_train_n_2, Y_train, epochs = 10, Visual = False)
Y_pred = sdca.predict(X_val_n)
error(Y_pred, Y_val)

# With standardized data
print("Kernel SVM regression with standardize data")
sdca = SDCARegressor(gauss_kernel, param = 0.001, C = 1)
sdca.fit(X_train_s_2, Y_train, epochs = 10, Visual = False)
Y_pred = sdca.predict(X_val_s)
error(Y_pred, Y_val)

# With standardized and orthogonalized data
print("Kernel SVM regression with standardized and orthogonalized data")
sdca = SDCARegressor(gauss_kernel, param = 0.001, C = 1)
sdca.fit(X_train_pca_2, Y_train, epochs = 10, Visual = False)
Y_pred = sdca.predict(X_val_pca)
error(Y_pred, Y_val)

# Testing handmade NN

print("Handmade NN")
layers = [m, 64, 1]
loss = LeastSquareCriterion()
mlp = MLP(layers, loss, lamb=0, delta=0.000000000001)
mlp.fit(X_train, Y_train, epochs=100, batch_size=512, Visual = True)
Y_pred = mlp.predict(X_val)
error(Y_pred, Y_val)

