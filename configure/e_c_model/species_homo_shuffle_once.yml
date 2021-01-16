dense_net_manager: &dense_net_manager
    name: 'dense_net_manager'
    dense_type: 'd121' #it can be d121,d169,d201 or d264 
    dense_k: 12
    conv_kernel_width: 3
    bottleneck_size: 1
    transition_pool_size: 2
    transition_pool_stride: 1
    theta: 1
    initial_conv_width: 3
    initial_stride: 1
    initial_filters: 48
    initial_pool_width: 2
    initial_pool_stride: 1
    use_global_pooling: False

basic_cnn_manager: &basic_cnn_manager
    #model manager name, currently support: basic_cnn_manager, dense_net_manager
    name: 'basic_cnn_manager'
    #embedding dimension
    embedding_dims: 16 
    #last hiddent layer width
    hidden_width: 256 
    #convolutional layer width
    conv_kernel_width: 3 
    #CNN layer length
    layer_len: 5 
    #convolutional layer length per CNN
    conv_len: 1 
    #the increae value of width of convolutional layer per CNN
    filter_delta: 24 
    #the pool size
    pool_size: 2 
    #the pool strides
    pooling_strides: 2 
    #the model saving name without suffix 
    save_model_name: 'E_C_model'
    #model saving path
    save_path: './models/'
    #optimizer name
    optimizer: 'Adam' 
    #loss function
    loss_function: 'binary_crossentropy'
    #last activation function
    last_activation: 'sigmoid'
    #early_stopping, only for single task currently
    early_stopping: True
    #after how many epoch the learning will stop if there is no improvement. effective only early_stopping takes effect 
    patience: 15 

data_config:
    #data manager name
    name: enzyme_data_manager
    #file path
    #where the training data is 
    file_path: uniprot-reviewed_yes_cluster_by_species.tab 
    #drop examples multi-labels more than the value. if 0, nothing will be dropped
    drop_multilabel: 0 
    #dummy label switch, if it is True. the uncertain labels like 3.2 will be deemed as a complete label I.E. 3.2.unknown.unknown
    apply_dummy_label: false 
    #max length of sequence
    max_len: 1000
    #print statistics
    print_statistics: True
    #how much portion of data will be used
    fraction: 1 
    #n-gram
    ngram: 1 
    #traning percentage, validation percentage will be 1- training percentage
    train_percent: 0.7
    #how many tasks will be 
    task_num: 4 
    #how many hieracical levels are there.
    level_num: 4 
    #when single task is setting, which level is the target level
    #Note:for single task only
    target_level: 4
    #label filed name in the file
    label_key: 'EC number'
    #id name
    id_name: 'Entry name'
    #if the examples of a class are below this value, that examples of the class will be removed
    class_example_threshhold: 1 
    #save data, if it is set, the data will be saved to files, the last one is the config file name, which save
    #object of python, and it will be saved with ending of .pkl
    save_data:
        train: train50_.tab
        test: test50_.tab
        meta: data_config50_

    #reuse:to reused data in files as training and testing
    #the last name is the config file name without .pkl
    #data_config['reuse_data'] = {'train':'train.tab', 'test':'test.tab', 'meta':'data_config'}
    #data_config['reuse_data'] = {'test':['cerevisiae.tab', 'rat.tab', 'mouse.tab', 'thaliana.tab'], 'train':['humap.tab']}
    reuse_data: 
        test: 
            - 'test_uniprot-reviewed_yes_cluster_by_species.tab'
        train:
            - 'train_uniprot-reviewed_yes_cluster_by_species.tab'

model_config: *basic_cnn_manager

evaluator_manager_config:
    print_summary: True
    #epochs
    epochs: 30 
    #batch_size
    batch_size: 200 
    #print report switch
    print_report: True
    #batch run:this function is in experiment
    batch_round: False 
    #batch run:this function is in experiment
    round_size: 1 
    #training model switch. when debug data, set to false, training will not be excuted
    train_model: True 
    #evaluator manager name: current support common_evaluator_manager
    name: 'common_evaluator_manager'
    #customized batch
    batch_generator: 
        {name: 'homogenous_cluster_training',
         #debug_file: 'debug_cluster.log',
         log_colums: ['Entry', 'Entry name', 'EC number', 'Cluster name'],
         cluster_col_name: 'Cluster name',
         ever_random: False 
        }
    #custom batch generator debug file
    debug_file: 'debug_file.tab'
