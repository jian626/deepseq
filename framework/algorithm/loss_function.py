import numpy as np
from sklearn.metrics import log_loss
from tensorflow.keras.losses import BinaryCrossentropy 
import tensorflow as tf
from tensorflow import math



def binary_entropy(y, y_, epsilon=1e-15):
    c_y_ = np.clip(y_, epsilon, 1. - epsilon)
    one_minus_y_ = np.clip(1-y_, epsilon, 1. - epsilon)
    return -1 * (np.sum(y * np.log(c_y_) + (1-y) * np.log(one_minus_y_), axis=1))


def get_custom_function(name):
    if 'weighted_binary_entropy' == name:
        return weighted_binary_entropy
    elif 'tensorflow_binary_entropy' == name:
        return tensorflow_binary_entropy

def tensorflow_binary_entropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0.0000001, 0.9999999)
    res = math.reduce_mean(-1 * (y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))) 
    return res

def weighted_binary_entropy(y_true_and_weights, y_pred):
    yws = tf.shape(y_true_and_weights)
    rn = yws[0] 
    cln =  yws[1] // 2
    y_true = tf.slice(y_true_and_weights, [0, 0], [rn, cln])
    weights = tf.slice(y_true_and_weights, [0, cln], [rn, cln])
    y_pred = tf.clip_by_value(y_pred, 0.0000001, 0.9999999)
    return math.reduce_mean(-1 * (y_true * math.log(y_pred) + (1-y_true) * math.log(1 - y_pred)) * weights)

if __name__ == '__main__':
    print('tensorflow_binary_entropy')
    y_true = tf.constant([[1., 0.] , [1., 1.]])
    y_pred = tf.constant([[0.9, 0.1], [0.1, 0.6]])
    bce = tf.keras.losses.BinaryCrossentropy()
    print(bce(y_true, y_pred).numpy())
    print(tensorflow_binary_entropy(y_true, y_pred).numpy())


