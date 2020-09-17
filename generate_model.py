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
import estimator_manager 
import data_manager 
import model_manager 
import estimator 

def run(input_data_config={}, input_model_config={}, input_estmator_config={}):
    transfor_learning= False
    utili.set_debug_flag(False)
    framdata_config = {}
    data_config = {}
    data_config['file_path'] = 'uniprot-reviewed_yes.tab'
    data_config['drop_multilabel'] = True 
    data_config['apply_dummy_label'] = False 
    data_config['max_len'] = 1000
    data_config['ec_level'] = 4 
    data_config['print_statistics'] = True
    data_config['fraction'] = 1 
    data_config['ngram'] = 2
    data_config['train_percent'] = 0.7
    data_config['task_num'] = 1 
    data_config['label_key'] = 'EC number'
    data_config['class_example_threshhold'] = 10 
    
    
    model_config = {}
    model_config['embedding_dims'] = 16 
    model_config['hidden1Dim'] = 256 
    model_config['hidden2Dim'] = 256 
    model_config['dense_net'] = False 
    model_config['cov_kernel_size'] = 3 
    model_config['layer_len'] = 1
    model_config['cov_len'] = 1
    model_config['filter_delta'] = 16
    model_config['pool_size'] = 2 
    model_config['pooling_strides'] = 1 
    model_config['save_model_name'] = 'my_model'
    model_config['save_path'] = './models/'

    estmator_config = {}
    estmator_config['print_summary'] = True
    estmator_config['optimizer'] = Adam()
    estmator_config['early_stopping'] = True
    estmator_config['patience'] = 20
    estmator_config['epochs'] = 20 
    estmator_config['batch_size'] = 400
    estmator_config['print_report'] = True
    estmator_config['batch_round'] = False 
    estmator_config['round_size'] = 1 

    for k in input_data_config:
        data_config[k] = input_data_config[k]

    for k in input_model_config:
        model_config[k] = input_model_config[k]

    for k in input_estmator_config:
        estmator_config[k] = input_estmator_config[k]

    dm = data_manager.enzyme_data_processor(data_config)
    x_train, y_train, x_test, y_test = dm.get_data(sep='\t')
    train_model = True 
    if train_model:
        mc = model_manager.model_creator(dm, model_config)
        mc.create_model()
        ee = estimator.enzyme_estimator(dm)
        estimator_list = []
        estimator_list.append(ee)
        me = estimator_manager.estimator_manager(estmator_config, dm, mc, estimator_list)
        me.evaluate()
        

if __name__ == '__main__':
    run()

    
    
    
    
    
