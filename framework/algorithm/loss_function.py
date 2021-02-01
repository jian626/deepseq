import numpy as np

def binary_entropy(y, y_, epsilon=1e-12):
    y = np.clip(y, epsilon, 1. - epsilon)
    y_ = np.clip(y_, epsilon, 1. - epsilon)
    return -1 * (np.sum(y * np.log(y_) + (1-y) * np.log(1-y_), axis=1))
