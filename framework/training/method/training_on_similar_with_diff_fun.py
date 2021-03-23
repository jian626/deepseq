from framework.training.method import training_method_creator as creator
from framework.algorithm import loss_function 
import numpy as np
import os
import sys
import copy
from framework.utili import get_table_value

class training_on_similar_with_diff_fun:
    name = 'training_on_similar_with_diff_fun'
    def __init__(self, config, context):
        print('*******************************training_on_similar_with_diff_fun************************************')
        print(config)
        self.config = config
        self.context = context
        log_file = get_table_value(self.config, 'debug_file', None)
        if log_file and os.path.exists(log_file):
              os.remove(log_file)
        train_set, _, _, _ = self.context['data_manager'].get_training_and_test_set()

        cluster_entry = get_table_value(config, 'cluster_entry', 'Cluster name')
        function_entry = get_table_value(config, 'function_entry', 'Function entry')
        print_statistics = get_table_value(config, 'print_statistics', True)
        
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

        print('===========================statistics for cluster===================================')
        print('differ_num:', differ_num)
        print('total:',len(train_set))
        print('rate:', differ_num / len(train_set))
        print('pairs num:', pair_num)

    def train(self):
        print('=======================train===============================')
        model_manager = self.context['model_manager']
        model = model_manager.get_model()
        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        data_manager = self.context['data_manager']
        task_num = data_manager.get_task_num() 
        x_train, y_train, _ = data_manager.get_training_data()
        for e_i in range(epochs):
            print('****************epochs %d****************' % e_i)
            id_pairs = copy.deepcopy(self.id_pairs)
            while len(id_pairs) > 0:
                print(len(id_pairs))
                batch = []
                need_del = []
                for cluster_name, items in id_pairs.items():
                    item = items.pop()
                    batch.append(item[0])
                    batch.append(item[1])
                    if len(items) == 0: 
                        need_del.append(cluster_name)

                    if len(batch) >= batch_size:
                        break

                for delete_name in need_del:
                    del id_pairs[delete_name]

                selected_x = x_train[batch]
                selected_y = []
                for i in range(task_num): 
                    selected_y.append(y_train[i][batch])
                model.train_on_batch(selected_x, selected_y)

def create(config, context):
    return training_on_similar_with_diff_fun(config, context)

creator.instance.register(training_on_similar_with_diff_fun.name, create)
