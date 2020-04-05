from pca import *
from linear_r import *

def sort_by_nearest(sample, X):
    """Sorting neighbors by smallest distance for every validation sample
    """
    indexes = [i for i in range(X_t.shape[0])]

    def distance(x):
        return np.linalg.norm(x[1] - sample)

    neigh = list(zip(indexes, X))

    neigh.sort(key = distance)

    sorted_ind = [n[0] for n in neigh]

    return sorted_ind

def create_NN(X_t, X_v):
    print("Creating nearest neighbors for every training sample")
    NN = []
    n_val_sample = len(X_v)
    for i in range(n_val_sample):
        x_v = X_v[i]
        print('Creating nearest neighbors, sample: '+str(i+1)+'/'+str(len(X_v))+'\r', sep=' ', end='', flush=True)
        NN.append(sort_by_nearest(x_v, X_t))

    return NN

X_s = standardize(X_train)
X_val_s = standardize(X_val)
NN = create_NN(X_s, X_val_s) # 1 minute of computing

def kNN_regression(NN, Y, k):
    Y_pred = []
    for n in NN:
        kNN = n[0:k]
        Y_pred.append((1/k) * sum(Y[nn] for nn in kNN)) # Mean of k nearest neighbors
    return Y_pred

Y_pred = kNN_regression(NN, Y_train, 10)

acc = accuracy(Y_pred, Y_val)

print("Error with kNN regression: " + str(acc))


def best_k_regression(NN, Y_t, Y_v):
    max_k = 100
    k_values = range(1, max_k)
    scores = []
    for k in k_values:
        print('Finding best k: '+str(k)+'/'+str(max_k)+'\r', sep=' ', end='', flush=True)
        scores.append(accuracy(kNN_regression(NN, Y_t, k), Y_v))
    print("\r")
    plt.figure(1)
    plt.plot(k_values, scores)
    plt.xlabel("Number nearest neighbors used")
    plt.ylabel("Error in %")
    plt.show()

    best_k = np.argmin(scores)
    print("best k: " + str(k_values[best_k]))
    return best_k
best_k = best_k_regression(NN, Y_train, Y_val)
