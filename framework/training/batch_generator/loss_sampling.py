from framework.training.batch_generator.training_base import training_base  
from framework.training.batch_generator import batch_generator_creator as creator
from framework.utili import get_table_value
import numpy as np
import random
import copy
class loss_sampling(training_base):
    name = 'loss_sampling'

    def __init__(self, config, context):
        self.context = context
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

    def get_train_examples(self):
        return self.train_examples

    def get_log_columns_names(self):
        return ['Entry', 'Entry name', 'EC number', 'Cluster name']

    def need_reset(self):
        return True 

    def get_debug_file_name(self):
        return None

    def get_binary_entropy(y, y_, epsilon=1e-12):
        y = np.clip(y, epsilon, 1. - epsilon)
        y_ = np.clip(y_, epsilon, 1. - epsilon)
        return -1 * (np.sum(y * np.log(y_) + (1-y) * np.log(1-y_), axis=1))

    def reset_samples(self):
        mm = self.evaluator_manager.get_model_manager()
        x, y = self.data_manager.get_training_data()
        model = mm.get_model()
        predicted = model.predict(x, y)[3]
        cross_np_loss = get_binary_entropy(y[3], predicted)
        indices = np.argsort(-cross_np_loss)
        self.train_examples = indices 

def create(config, context):
    return loss_sampling(config, context)

creator.instance.register(loss_sampling.name, create)
