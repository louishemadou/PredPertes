import math
import matplotlib.pyplot as plt
import numpy as np

def gen_batchs(data, labels, batch_size):
    """Créé des batchs de taille batch_size
    de data et labels"""
    assert(len(data) == len(labels))
    r = np.random.permutation(len(data))
    rdata = data[r[:]]
    rlabels = labels[r[:]]
    for i in range(0, len(data), batch_size):
        yield rdata[i:i+batch_size], rlabels[i:i+batch_size]


class Module(object):
    def __init__(self):
        self.gradInput = None
        self.output = None

    def forward(self, *input):
        """Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *input):
        """Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError


class LeastSquareCriterion(Module):

    def __init__(self):
        super(LeastSquareCriterion, self).__init__()

    def forward(self, x, y):
        self.output = np.sum((x-y)**2)
        return np.sum(self.output)

    def backward(self, x, y):
        y.shape = x.shape
        self.gradInput = 2*(x-y)
        return self.gradInput


class Linear(Module):
    """
    The input is supposed to have two dimensions (batchSize,in_feature)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = math.sqrt(
            1. / (out_features * in_features))*np.random.randn(out_features, in_features)
        self.bias = np.zeros(out_features)
        self.gradWeight = None
        self.gradBias = None

    def forward(self, x):
        self.output = np.dot(x, self.weight.transpose(
        ))+np.repeat(self.bias.reshape([1, -1]), x.shape[0], axis=0)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = np.dot(gradOutput, self.weight)
        self.gradWeight = np.dot(gradOutput.transpose(), x)
        self.gradBias = np.sum(gradOutput, axis=0)
        return self.gradInput

    def gradientStep(self, lr, lamb):
        self.weight = (1-lamb)*self.weight-lr*self.gradWeight
        self.bias = self.bias-lr*self.gradBias


class ReLU(Module):

    def __init__(self, bias=True):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.output = x.clip(0)
        return self.output

    def backward(self, x, gradOutput):
        self.gradInput = (x > 0)*gradOutput
        return self.gradInput

class MLP(Module):
    """On donne en entrée une liste de tailles
    de couches intermédiaires souhaitées."""
    def __init__(self, layers, loss, lamb = 0.001, delta = 0.01):
        super(MLP, self).__init__()
        self.loss = loss
        self.lamb = lamb # Regularization
        self.delta = delta # learning rate

        self.n_layers = len(layers)
       
        self.fc = [] # Fully connected layers

        for i in range(self.n_layers-1): # Couches cachées
            self.fc.append(Linear(layers[i], layers[i+1]))
            
        self.n_fc = len(self.fc)
        self.relu = [ReLU() for _ in range(self.n_fc-1)]
        
    def forward(self, x):
        for i in range(self.n_fc-1):
            x = self.fc[i].forward(x)
            x = self.relu[i].forward(x)
        x = self.fc[-1].forward(x)
        return x
    
    def backward(self, x, gradient):
        for i in range(self.n_fc-1, 0, -1):
            gradient = self.fc[i].backward(self.relu[i-1].output, gradient)
            gradient = self.relu[i-1].backward(self.fc[i-1].output, gradient)
        gradient = self.fc[0].backward(x, gradient)
        return gradient
    
    def gradientStep(self,lr,lamb):
        for i in range(self.n_fc-1, -1, -1):
            self.fc[i].gradientStep(lr,lamb) 
        return True
    
    def fit(self, X, Y, epochs = 100, batch_size = 16, Visual = False):
        loss_values_train = []
        n_train = len(X)
        for k in range(epochs):
            print('training, iteration: '+str(k+1)+'/'+str(epochs)+'\r', sep=' ', end='', flush=True)
            batchs = gen_batchs(X, Y, batch_size) # Création des batchs pour l'epoch k
            for batch_data in batchs:
                batch = batch_data[0]
                batch_labels = batch_data[1]
                gradInput = self.loss.backward(self.forward(batch), batch_labels)
                self.backward(batch, gradInput)
                self.gradientStep(self.delta, self.lamb)
            if Visual: # Calcul de l'erreur quadratique moyenne pour les données d'entrainement et de validation
                loss_values_train.append((1/n_train)*self.loss.forward(self.forward(X), Y))
            
        if Visual:
            it = range(len(loss_values_train))
            plt.figure()
            plt.plot(it, loss_values_train, 'r', label = "Training loss")
            plt.legend()
            plt.title("Loss over epochs")
            plt.show()

    def predict(self, X):
        return self.forward(X)

    def NbParameters(self):
        Nb = sum((self.fc[i].in_features + 1)*self.fc[i].out_features for i in range(self.n_fc))
        print(str(Nb) + " paramètres")
