from framework.training.batch_generator.training_base import training_base  
from framework.training.batch_generator import batch_generator_creator as creator
from framework.utili import get_table_value
import numpy as np
import random
import copy
class training_simi_diff_fun:
    name = 'training_simi_diff_fun'

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

        cluster_training_base.__init__(self, config, context)

        train_set, _, _, _ = self.context['data_manager'].get_training_and_test_set()

        cluster_entry = get_table_value(config,'cluster_col_name', 'Cluster name')
        function_entry = get_table_value(config, 'function_entry', 'Function entry')
        print_statistics = get_table_value(config, 'print_statistics', True)
        self.mix_num = get_table_value(config, 'mix_num', 0)
    
        
        total_clusters = {}
        differ_clusters = set() 
        differ_num = 0
        for index in range(len(train_set)):
            row = train_set.iloc[index]
            cluster_name = row[cluster_entry]
            cluster = None
            if not cluster_name in total_clusters:
                total_clusters[cluster_name] = {} 
            cluster = total_clusters[cluster_name]    
            num_cluster_before = len(cluster)
            function_str = str(row[function_entry]).strip()
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
        id_pairs = {} 
        pair_num = 0
        for cluster_name in differ_clusters:
            cluster = total_clusters[cluster_name]
            pairs = []
            values = list(cluster.values())
            for i in range(len(values)):
                for item_a in values[i]:
                    for j in range(i, len(values)):
                        for item_b in values[j]:
                            pairs.append((item_a, item_b))
                            pair_num += 1
            id_pairs[cluster_name] = pairs
        self.id_pairs = id_pairs
        print('**********************************id_pairs**********************:', len(id_pairs))
        self.batch_num = int(pair_num * 2 / self.batch_size)
        self.reset_samples()

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
        id_pairs = copy.deepcopy(self.id_pairs)
        train_examples = []
        while len(id_pairs) > 0:
            need_del = []
            for cluster_name, items in id_pairs.items():
                item = items.pop()
                train_examples.append(item[0])
                train_examples.append(item[1])
                if len(items) == 0: 
                    need_del.append(cluster_name)

            for delete_name in need_del:
                del id_pairs[delete_name]

        self.train_examples = train_examples

    def __len__(self):
        return self.batch_num 

def create(config, context):
    return training_simi_diff_fun(config, context)

creator.instance.register(training_simi_diff_fun.name, create)
