from datetime import datetime
from framework import init 
from framework import utili
from framework.evaluator_manager import evaluator_manager_creator 
from framework.model_manager import model_manager_creator
from framework.evaluator import evaluator_creator 
from framework.data_manager import data_manager_creator 
import sys

def run(file_path=None, input_data_config={}, input_model_config={}, input_evaluator_manager_config={}):
    transfor_learning= False
    utili.set_debug_flag(False)
    framdata_config = {}
    data_config = {}
    data_config['name'] = 'enzyme_protein_data_manager'
    if file_path is None:
        file_path = 'uniprot-reviewed_yes.tab'
    #the path of data used
    data_config['file_path'] = file_path
    #max length of sequence, the sequence beyond that will be eliminated
    data_config['max_len'] = 1000
    #print statistics switch
    data_config['print_statistics'] = True
    #the used proportion of data 
    data_config['fraction'] = 1
    #ngram
    data_config['ngram'] = 1 
    #training percentage
    data_config['train_percent'] = 0.7
    #currently only 1 is supported for enzyme protein classifier generator
    data_config['task_num'] = 1 
    
    model_config = {}
    #embedding dimension 
    model_config['embedding_dims'] = 16
    #the last hidden layer width 
    model_config['hidden_width'] = 256
    #convolutional layer kernel width
    model_config['conv_kernel_width'] = 3
    #convolutional layer strides 
    model_config['conv_strides'] = 1
    #how many CNN layers 
    model_config['layer_len'] = 1
    #how many convolutional layers per CNN layer 
    model_config['conv_len'] = 1
    #the increase value of each convolutional layer relative to previous convolutional layer
    model_config['filter_delta'] = 16
    #pooling size
    model_config['pool_size'] = 16
    #pooling strides 
    model_config['pooling_strides'] = 16
    #the generated model name without 
    model_config['save_model_name'] = 'E_P_model'
    #where the model will be generated
    model_config['save_path'] = './models/'
    #last activation
    model_config['last_activation'] = 'softmax'
    #losst function
    model_config['loss_function'] = 'categorical_crossentropy'
    #optimizer name
    model_config['optimizer'] = 'Adam'
    #Main part manager name, support basic_cnn_manager, dense_net_manager
    model_config['name'] = 'basic_cnn_manager'
    #following commented out configuration is for  dense_net 
    '''
    model_config['name'] = 'dense_net_manager'
    model_config['dense_type'] = 'd121' #if dense_net_manager is used, it can be d121,d169,d201 or d264 
    model_config['dense_k'] = 12#dense_net configurations
    model_config['conv_kernel_width'] = 3
    model_config['bottleneck_size'] = 1
    model_config['transition_pool_size'] = 2
    model_config['transition_pool_stride'] = 2 
    model_config['theta'] = 1
    model_config['initial_conv_width'] = 3
    model_config['initial_stride'] = 1
    model_config['initial_filters'] = 12 
    model_config['initial_pool_width'] = 2
    model_config['initial_pool_stride'] = 2 
    model_config['use_global_pooling'] = False
    '''

    evaluator_manager_config = {}
    #summary result switch
    evaluator_manager_config['print_summary'] = True
    #early_stopping, only for single task currently
    evaluator_manager_config['early_stopping'] = True
    #epochs
    evaluator_manager_config['epochs'] = 30
    #batch size
    evaluator_manager_config['batch_size'] = 200
    #print report swith
    evaluator_manager_config['print_report'] = True
    #this function is in experiment 
    evaluator_manager_config['batch_round'] = False
    #this function is in experiment 
    evaluator_manager_config['round_size'] = 1
    #evaluator manager name: currently only common_evaluator_manager
    evaluator_manager_config['name'] = 'common_evaluator_manager'
    #need to train model or just print some information 
    evaluator_manager_config['train_model'] = True

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
    ee = evaluator_creator.instance.create('enzyme_protein_evaluator',dm)
    evaluator_list = []
    evaluator_list.append(ee)
    me = evaluator_manager_creator.instance.create(evaluator_manager_config, dm, mc, evaluator_list)
    me.evaluate()

if __name__ == '__main__':
    file_path = None
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    run(file_path)
