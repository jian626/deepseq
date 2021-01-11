from tensorflow.keras.utils import Sequence
import numpy as np
import random
import copy
from framework.utili import get_table_value
class training_base(Sequence):
    def reset_samples(self):
        pass

    def get_train_examples():
        pass

    def get_log_columns_names():
        pass

    def need_reset(self):
        pass

    def get_debug_file_name(self):
        pass

    def __getitem__(self, index):
        begin = index * self.batch_size
        end = begin + self.batch_size
        result = self.get_train_examples()[begin:end]
        training_set, _ = self.data_manager.get_training_and_test_set()
        debug_file = self.get_debug_file_name()
        if debug_file:
            columns_names = self.get_log_columns_names()
            debug_df = training_set.iloc[result]
            if columns_names:
                debug_df = debug_df[columns_names]
            debug_df.to_csv(debug_file, sep='\t', index=False, mode='a')
        x, y = self.data_manager.get_training_data()
        rx = x[result]
        ry = []
        task_num = self.data_manager.get_task_num()
        for i in range(task_num):
            ry.append(y[i][result])
        return rx, ry

    def on_epoch_end(self):
        if self.need_reset():
            self.reset_samples()
        self.debug_file = None

class cluster_training_base(training_base):
    def __init__(self, data_manager, batch_size, config):
        cluster_col_name = get_table_value(config,'cluster_col_name', 'Cluster name')
        ever_random = get_table_value(config, 'ever_random', False)
        debug_file = get_table_value(config, 'debug_file')
        log_colums = get_table_value(config, 'log_colums')

        self.data_manager = data_manager
        self.batch_size = batch_size
        self.log_colums = log_colums 
        self.cluster_col_name = cluster_col_name
        x, y = self.data_manager.get_training_data()
        self.sample_len = None 
        self.cluster_info = None
        self.sample_len_store = len(x)
        print('sample_len:', self.sample_len)
        self.batch_num = int(np.floor(len(x) / self.batch_size))
        self.cluster_info_store = {} 
        self.debug_file = debug_file
        training_set, _ = self.data_manager.get_training_and_test_set()
        print('Cluster name')
        for i in range(training_set.shape[0]):
            cluster_name = training_set.iloc[i][self.cluster_col_name]
            cluster_members = None
            if not cluster_name in self.cluster_info_store:
                self.cluster_info_store[cluster_name] = []
            cluster_members = self.cluster_info_store[cluster_name]
            cluster_members.append(i)
        self.cluster_keys = None 
        self.reset_samples()

    def get_log_columns_names():
        return self.log_colums

    def need_reset(self):
        return self.ever_random

    def get_debug_file_name(self):
        return self.debug_file

    def get_train_examples():
        return self.train_examples

    def __len__(self):
        return self.batch_num 

    def reset_samples(self):
        pass


