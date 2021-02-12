import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy 



def binary_entropy(y, y_, epsilon=1e-12):
    bce = BinaryCrossentropy(from_logits=True)
    return bce(y, y_).numpy()
    #c_y_ = np.clip(y_, epsilon, 1. - epsilon)
    #one_minus_y_ = np.clip(1-y_, epsilon, 1. - epsilon)
    #return -1 * (np.sum(y * np.log(c_y_) + (1-y) * np.log(1-y_), axis=1))
