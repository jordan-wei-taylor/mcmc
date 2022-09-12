from   mcmc.utils import sigmoid, softmax
from   scipy import stats

import numpy as np

def log_pstar_classification(theta, data, **kwargs):
    X, y = data
    p    = len(np.unique(y))
    bw   = theta.reshape(-1, p)
    b, w = bw[0], bw[1:]
    hat  = softmax(X @ w + b)
    return np.log(hat[range(len(hat)), y]).sum()

def log_pstar_binary_classification(theta, data, **kwargs):
    X, y = data
    bw   = theta.reshape(X.shape[1] + 1)
    b, w = bw[0], bw[1:]
    hat  = sigmoid(X @ w + b)
    return np.log(hat[y == 1]).sum() - np.log(1 - hat[y == 0]).sum()

def log_pstar_regression(theta, data, noise_deviation = 0.2, **kwargs):
    X, y = data
    b, w = theta[0], theta[1:]
    return stats.norm(X @ w + b, noise_deviation).logpdf(y).sum()