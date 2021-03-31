from tensorflow.keras.utils import Sequence
import numpy as np
import random
import copy
from framework.utili import get_table_value
class training_base(Sequence):
    def __init__(self, config=None, context=None):
        if context:
            self.context = context
            self.data_manager = self.context['data_manager']
            self.ever_random = get_table_value(config, 'ever_random', False)
            self.debug_file = get_table_value(config, 'debug_file')
            self.log_colums = get_table_value(config, 'log_colums')
            self.batch_size = get_table_value(config, 'batch_size')
            self.mix_num = get_table_value(config, 'mix_num', 0)
            self.mix_method = get_table_value(config, 'mix_method', None)
            if self.mix_method and self.mix_num > 0:
                if self.mix_method == 'pair':
                    self.cluster_col_name = get_table_value(config,'cluster_col_name', 'Cluster name')
                    self.function_entry = get_table_value(config, 'function_entry', 'EC number')
                    self.get_mix_pairs() 

    def reset_samples(self):
        pass

    def get_train_examples(self):
        pass

    def get_log_columns_names(self):
        pass

    def need_reset(self):
        pass

    def get_debug_file_name(self):
        pass

    def __getitem__(self, index):
        begin = index * self.batch_size
        end = begin + self.batch_size
        result = self.get_train_examples()[begin:end]
        if self.mix_num and self.mix_num > 0: 
            print('mix_method:', self.mix_method)
            if self.mix_method == 'pair':
                if self.id_pairs:
                    pairs = random.sample(self.id_pairs, self.mix_num // 2)
                    for pair in pairs:
                        result += pair
            else:
                result += random.sample(self.get_train_examples(), self.mix_num)
            
        training_set, _, _, _ = self.data_manager.get_training_and_test_set()
        debug_file = self.get_debug_file_name()
        if debug_file:
            columns_names = self.get_log_columns_names()
            debug_df = training_set.iloc[result]
            if columns_names:
                debug_df = debug_df[columns_names]
            debug_df.to_csv(debug_file, sep='\t', index=False, mode='a')
        x, y, _ = self.data_manager.get_training_data()
        rx = x[result]
        ry = []
        task_num = self.data_manager.get_task_num()
        for i in range(task_num):
            ry.append(y[i][result])
        return rx, ry


    def get_mix_pairs(self):
        train_set, _, _, _ = self.data_manager.get_training_and_test_set()
        total_clusters = {}
        differ_clusters = set() 
        differ_num = 0

        for index in range(len(train_set)):
            row = train_set.iloc[index]
            cluster_name = row[self.cluster_col_name]
            cluster = None
            if not cluster_name in total_clusters:
                total_clusters[cluster_name] = {} 
            cluster = total_clusters[cluster_name]    
            num_cluster_before = len(cluster)
            function_str = str(row[self.function_entry]).strip()
            if not function_str in cluster:
                cluster[function_str] = []
            num_cluster_after = len(cluster)
            function_group = cluster[function_str]
            function_group.append(index)
            if num_cluster_before == 1 and num_cluster_after == 2:
                differ_num += 1
                differ_clusters.add(cluster_name)
        self.total_clusters = total_clusters
        self.differ_clusters = differ_clusters
        id_pairs = [] 
        pair_num = 0
        for cluster_name in differ_clusters:
            cluster = total_clusters[cluster_name]
            values = list(cluster.values())
            for i in range(len(values)):
                for item_a in values[i]:
                    for j in range(i, len(values)):
                        for item_b in values[j]:
                            id_pairs.append([item_a, item_b])
                            pair_num += 1
        random.shuffle(id_pairs)
        self.id_pairs = id_pairs
        print('**********************************id_pairs**********************:', len(id_pairs))

    def on_epoch_end(self):
        if self.need_reset():
            self.reset_samples()
        self.debug_file = None

class cluster_training_base(training_base):
    def __init__(self, config, context):
        super(cluster_training_base, self).__init__(config, context)
        cluster_col_name = get_table_value(config,'cluster_col_name', 'Cluster name')

        self.function_entry = get_table_value(config, 'function_entry', 'EC number')
        self.cluster_col_name = cluster_col_name
        x, y_, _ = self.data_manager.get_training_data()
        self.sample_len = None 
        self.cluster_info = None
        self.sample_len_store = len(x)
        print('sample_len:', self.sample_len)
        self.batch_num = int(np.floor(len(x) / self.batch_size))
        self.cluster_info_store = {} 
        training_set, _, _, _ = self.data_manager.get_training_and_test_set()
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


    def get_log_columns_names(self):
        return self.log_colums

    def need_reset(self):
        return self.ever_random

    def get_debug_file_name(self):
        return self.debug_file

    def get_train_examples(self):
        return self.train_examples

    def __len__(self):
        print('__len__:', self.batch_num)
        return self.batch_num 

    def reset_samples(self):
        pass


