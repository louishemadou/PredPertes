import keras
import tensorflow as tf
import os
from pca import *
from utils import error
from split import retrieve_and_split, retrieve_all_and_split
from visualize import compare

os.environ['TF_KERAS'] = '1'

X_train, Y_train, X_val, Y_val, X_test, Y_test = retrieve_and_split(2018)

# Testing PCA and creating modified sets

m = X_train.shape[1]

X_train_n = normalize(X_train)
X_val_n = normalize(X_val)

X_train_s = standardize(X_train)
X_val_s = standardize(X_val)

X_train_pca = pca(X_train_s, m, whitening = True, visual = False)
X_val_pca = pca(X_val_s, m, whitening = True, visual = False)

# Testing Linear_regression


#Création du model (layers)
model = keras.models.Sequential()
model.add(keras.layers.Dense(400, activation='sigmoid', input_dim=m))
model.add(keras.layers.Dense(400, activation='sigmoid'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

# Optimizer

SDG = keras.optimizers.SGD(learning_rate = 0.00001)

# Compile model

model.compile(loss='mean_squared_error', optimizer=SDG, metrics=[keras.metrics.MeanSquaredError()])

# Fit the model
history = model.fit(X_train_s, Y_train, epochs=50, batch_size=100)

# summarize history for accuracy
plt.plot(history.history['loss'])
plt.title('loss over epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
# calculate predictions
Y_pred = model.predict(X_train_s)

error(Y_pred, Y_train)
compare(Y_val, Y_pred)
