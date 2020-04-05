from pca import *

# Linear regression

def add_ones(X):
    new_X = np.array([np.ones(X.shape[0])] + [sample for sample in X.T])
    return new_X.T

def grad_error(X, Y, beta):
    """Computes gradient of the loss
    function"""
    n_sample = X.shape[0]
    Y_pred = np.dot(X, beta)
    dbeta = 2 * (np.sum( [(Y_pred[i]-Y[i]) * X[i] for i in range(n_sample)], axis=0))
    return dbeta

n_comp = 35
X_pca = add_ones(pca(standardize(X_train), n_comp))
X_val_pca = add_ones(pca(standardize(X_val), n_comp))

def linear_regression(X, Y, n_iter = 50, delta = 0.0001, ridge = 0, lasso = 0):
    n_feature = X.shape[1]
    
    beta = np.zeros(n_feature)
    
    for i in range(n_iter): # Gradient algorithm
        dbeta = grad_error(X, Y, beta)
        dbeta += 2*ridge*beta
        dbeta += lasso * (2*(beta > 0).astype(int) - 1)
        beta = beta - delta * dbeta
        #print(np.linalg.norm(dbeta))
        print('training, iteration: '+str(i+1)+'/'+str(n_iter)+'\r', sep=' ', end='', flush=True)
    print("\r")
    return beta

def accuracy(Y_pred, Y):
    """Measures accuracy of a
    regression prediction"""
    error = np.abs(Y_pred - Y)
    return 100*np.mean(error)/np.mean(Y) # Mean experimental error  


beta = linear_regression(X_pca, Y_train, n_iter = 100)
Y_pred = np.dot(X_val_pca, beta)

acc = accuracy(Y_pred, Y_val)
print("Error with gradient algorithm used: " + str(acc) + "%")

