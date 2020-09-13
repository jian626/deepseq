import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization, AveragePooling1D
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import utili
import BioDefine
from datetime import datetime
from sklearn.metrics import classification_report
import framework

if __name__ == '__main__':

    framdata_config = {}
    data_config = {}
    data_config['file_path'] = 'uniprot-reviewed_yes.tab'
    data_config['drop_multilabel'] = False
    data_config['apply_dummy_label'] = False 
    data_config['max_len'] = 1000
    data_config['ec_level'] = 4
    data_config['print_statistics'] = True
    data_config['fraction'] = 1 
    data_config['ngram'] = 2
    data_config['train_percent'] = 0.7
    
    model_config = {}
    model_config['embedding_dims'] = 16 
    model_config['hidden1Dim'] = 256 
    model_config['hidden2Dim'] = 256 
    model_config['multi_task'] = False 
    model_config['dense_net'] = True
    model_config['cov_kernel_size'] = 3 
    model_config['filter_delta'] = 16
    model_config['pool_size'] = 2
    model_config['pooling_strides'] = 1
    model_config['save_model_name'] = 'model.h5'

    dp = framework.data_processor(data_config)
    x_train, y_train, x_test, y_test = dp.get_data()
    mc = framework.model_creator(data_config, model_config)
    mc.create_model()
    estmator_config = {}
    estmator_config['print_summary'] = True
    estmator_config['optimizer'] = Adam()
    estmator_config['early_stopping'] = True
    estmator_config['patience'] = 20
    estmator_config['epochs'] = 1 
    estmator_config['batch_size'] = 400
    estmator_config['print_report'] = True
    me = framework.model_estimator(estmator_config, dp, mc)
    me.evaluate()
    
    
    
    
    
