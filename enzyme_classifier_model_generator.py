import numpy as np
import pandas as pd
from framework import utili
from datetime import datetime
from sklearn.metrics import classification_report
from framework.estimator_manager import estimator_manager_creator 
from framework.data_manager import data_manager_creator 
from framework.model_manager import model_manager_creator
from framework.estimator import estimator_creator

def run(input_data_config={}, input_model_config={}, input_estimator_manager_config={}):
    transfor_learning= False
    utili.set_debug_flag(False)
    framdata_config = {}
    data_config = {}
    data_config['name'] = 'enzyme_data_manager'
    data_config['file_path'] = 'uniprot-reviewed_yes.tab'
    data_config['drop_multilabel'] = True 
    data_config['apply_dummy_label'] = False 
    data_config['max_len'] = 1000
    data_config['ec_level'] = 4 
    data_config['print_statistics'] = True
    data_config['fraction'] = 1 
    data_config['ngram'] = 3 
    data_config['train_percent'] = 0.7
    data_config['task_num'] = 4 #currently only 1 or 4 is supported for enzyme classifier generator
    data_config['label_key'] = 'EC number'
    data_config['class_example_threshhold'] = 10 
    
    model_config = {}
    model_config['embedding_dims'] = 16 
    model_config['hidden_width'] = 256 
    model_config['dense_net'] = False 
    model_config['cov_kernel_size'] = 3 
    model_config['layer_len'] = 1 
    model_config['cov_len'] = 1 
    model_config['filter_delta'] = 1
    model_config['pool_size'] = 2 
    model_config['pooling_strides'] = 1 #when it is dense net, only strides 1 is currently supported
    model_config['save_model_name'] = 'enzyme_model'
    model_config['save_path'] = './models/'
    model_config['optimizer'] = 'Adam' 
    model_config['loss_function'] = 'binary_crossentropy'
    model_config['early_stopping'] = False
    model_config['last_activation'] = 'sigmoid'


    estimator_manager_config = {}
    estimator_manager_config['print_summary'] = True
    estimator_manager_config['early_stopping'] = True
    estimator_manager_config['patience'] = 20
    estimator_manager_config['epochs'] = 200 
    estimator_manager_config['batch_size'] = 400
    estimator_manager_config['print_report'] = True
    estimator_manager_config['batch_round'] = False 
    estimator_manager_config['round_size'] = 1 
    estimator_manager_config['train_model'] = True

    for k in input_data_config:
        data_config[k] = input_data_config[k]

    for k in input_model_config:
        model_config[k] = input_model_config[k]

    for k in input_estimator_manager_config:
        estimator_manager_config[k] = input_estimator_manager_config[k]

    dm = data_manager_creator.create(data_config)
    x_train, y_train, x_test, y_test = dm.get_data(sep='\t')
    mc = model_manager_creator.create(dm, model_config)
    mc.create_model()
    ee = estimator_creator.create('enzyme_estimator',dm)
    estimator_list = []
    estimator_list.append(ee)
    me = estimator_manager_creator.create(estimator_manager_config, dm, mc, estimator_list)
    me.evaluate()

if __name__ == '__main__':
    run()
    
    
