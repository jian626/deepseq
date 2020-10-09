import numpy as np
import pandas as pd
import sys
from framework import init 
from framework import utili
from datetime import datetime
from sklearn.metrics import classification_report
from framework.evaluator_manager import evaluator_manager_creator 
from framework.data_manager import data_manager_creator 
from framework.model_manager import model_manager_creator
from framework.evaluator import evaluator_creator

def run(file_path=None, input_data_config={}, input_model_config={}, input_evaluator_manager_config={}):
    transfor_learning= False
    utili.set_debug_flag(False)
    framdata_config = {}
    data_config = {}
    #the data manager name
    data_config['name'] = 'enzyme_data_manager'
    if file_path is None:
        file_path = 'uniprot-reviewed_yes.tab'
    #where the training data is 
    data_config['file_path'] = file_path
    #drop multi-label?
    data_config['drop_multilabel'] = False 
    #dummy label switch
    data_config['apply_dummy_label'] = False 
    #max length of sequence
    data_config['max_len'] = 1000
    #print statistics
    data_config['print_statistics'] = True
    #how much portion of data will be used
    data_config['fraction'] = 1 
    #n-gram
    data_config['ngram'] = 1 
    #traning percentage, validation percentage will be 1- training percentage
    data_config['train_percent'] = 0.7
    #how many tasks will be 
    data_config['task_num'] = 4 
    #how many hieracical levels are there.
    data_config['level_num'] = 4 
    #when single task is setting, which level is the target level
    data_config['target_level'] = 4#for single task only
    #label filed name in the file
    data_config['label_key'] = 'EC number'
    #if the examples of a class are below this value, that examples of the class will be removed
    data_config['class_example_threshhold'] = 0 
    
    model_config = {}
    #embedding dimension
    model_config['embedding_dims'] = 16 
    #last hiddent layer width
    model_config['hidden_width'] = 256 
    #convolutional layer width
    model_config['conv_kernel_width'] = 3 
    #CNN layer length
    model_config['layer_len'] = 1 
    #convolutional layer length per CNN
    model_config['conv_len'] = 1 
    #the increae value of width of convolutional layer per CNN
    model_config['filter_delta'] = 1
    #the pool size
    model_config['pool_size'] = 16 
    #the pool strides
    model_config['pooling_strides'] = 16 
    #the model saving name without suffix 
    model_config['save_model_name'] = 'E_C_model'
    #model saving path
    model_config['save_path'] = './models/'
    #optimizer name
    model_config['optimizer'] = 'Adam' 
    #loss function
    model_config['loss_function'] = 'binary_crossentropy'
    #activation function
    model_config['last_activation'] = 'sigmoid'
    #model manager name, currently support: basic_cnn_manager, dense_net_manager
    model_config['name'] = 'basic_cnn_manager'
    #early_stopping, only for single task currently
    model_config['early_stopping'] = True
    #after how many epoch the learning will stop if there is no improvement. effective only early_stopping takes effect 
    model_config['patience'] = 40

    #the following configuration commented out is for dense_net_manager
    '''
    model_config['name'] = 'dense_net_manager'
    model_config['dense_type'] = 'd121' #it can be d121,d169,d201 or d264 
    model_config['dense_k'] = 12
    model_config['conv_kernel_width'] = 3
    model_config['bottleneck_size'] = 1
    model_config['transition_pool_size'] = 2
    model_config['transition_pool_stride'] = 1
    model_config['theta'] = 1
    model_config['initial_conv_width'] = 3
    model_config['initial_stride'] = 1
    model_config['initial_filters'] = 48
    model_config['initial_pool_width'] = 2
    model_config['initial_pool_stride'] = 1
    model_config['use_global_pooling'] = False
    '''


    evaluator_manager_config = {}
    #print summary switch 
    evaluator_manager_config['print_summary'] = True
    #epochs
    evaluator_manager_config['epochs'] = 30 
    #batch_size
    evaluator_manager_config['batch_size'] = 200 
    #print report switch
    evaluator_manager_config['print_report'] = True
    #batch run:this function is in experiment
    evaluator_manager_config['batch_round'] = False 
    #batch run:this function is in experiment
    evaluator_manager_config['round_size'] = 1 
    #training model switch. when debug data, set to false, training will not be excuted
    evaluator_manager_config['train_model'] = True 
    #evaluator manager name: current support common_evaluator_manager
    evaluator_manager_config['name'] = 'common_evaluator_manager'

    for k in input_data_config:
        data_config[k] = input_data_config[k]

    for k in input_model_config:
        model_config[k] = input_model_config[k]

    for k in input_evaluator_manager_config:
        evaluator_manager_config[k] = input_evaluator_manager_config[k]

    dm = data_manager_creator.instance.create(data_config)
    x_train, y_train, x_test, y_test = dm.get_data(sep='\t')
    mc = model_manager_creator.instance.create(dm, model_config)
    mc.create_model()
    #the evaluator name should be given: 
    #currently,  E_P enzyme_protein_evaluator should be used
    #            E_C enzyme_evaluator should be used
    ee = evaluator_creator.instance.create('enzyme_evaluator',dm)
    evaluator_list = []
    evaluator_list.append(ee)
    me = evaluator_manager_creator.instance.create(evaluator_manager_config, dm, mc, evaluator_list)
    me.evaluate()

if __name__ == '__main__':
    file_path = None
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    run(file_path)
    
