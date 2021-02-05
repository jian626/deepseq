from framework.training.batch_generator.training_base import training_base  
from framework.training.batch_generator import batch_generator_creator as creator
from framework.utili import get_table_value
from framework.algorithm import loss_function 
import numpy as np
import random
import copy
class loss_sampling(training_base):
    name = 'loss_sampling'

    def __init__(self, config, context):
        self.context = context
        self.config = config
        cluster_col_name = get_table_value(config,'cluster_col_name', 'Cluster name')
        ever_random = get_table_value(config, 'ever_random', False)
        debug_file = get_table_value(config, 'debug_file')
        log_colums = get_table_value(config, 'log_colums')
        batch_size = get_table_value(config, 'batch_size')

        self.ever_random = ever_random
        data_manager = self.context['data_manager']
        model_manager = self.context['model_manager']
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.log_colums = log_colums 
        self.reset_samples()
        x, y = self.data_manager.get_training_data()
        self.batch_num = int(np.floor(len(x) / self.batch_size))

    def get_train_examples(self):
        return self.train_examples

    def get_log_columns_names(self):
        return ['Entry', 'Entry name', 'EC number', 'Cluster name']

    def need_reset(self):
        return True 

    def get_debug_file_name(self):
        debug_file = get_table_value(self.config,'debug_file', None)
        return debug_file 

    def reset_samples(self):
        test_file = open('test_log.txt', 'a')
        print('---------------------------reset_samples--------------------------------')
        mm = self.model_manager
        x, y = self.data_manager.get_training_data()
        model = mm.get_model()
        predicted = model.predict(x)[3]
        loss = loss_function.binary_entropy(y[3], predicted)
        indices = None
        hard_first = get_table_value(self.config, 'hard_first', False) 
        sampling_with_replace = get_table_value(self.config, 'sampling_with_replace', None)
        if not sampling_with_replace is None:
            n = len(predicted)
            if hard_first:
                p = loss /np.sum(loss)
            else:
                p = 1 - loss /np.sum(loss)
            #indices = np.random.choice(a=[x for x in range(n)], size=n, replace=sampling_with_replace, p=p)
            indices = random.choices([x for x in range(n)], weights=p, k=n)
            test_file.writelines(str(indices))
        test_file.close()
        else:
            if hard_first:
                indices = np.argsort(-loss)
            else:
                indices = np.argsort(loss)
        self.train_examples = indices 

    def __len__(self):
        return self.batch_num 

def create(config, context):
    return loss_sampling(config, context)

creator.instance.register(loss_sampling.name, create)
