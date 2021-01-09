from tensorflow.keras.utils import Sequence
import numpy as np
import random
import copy
class SequenceGenerator(Sequence):
    def __init__(self, data_manager, batch_size, ever_random=True, debug_file = None):
        self.data_manager = data_manager
        self.batch_size = batch_size
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
            cluster_name = training_set.iloc[i]['Cluster name']
            cluster_members = None
            if not cluster_name in self.cluster_info_store:
                self.cluster_info_store[cluster_name] = []
            cluster_members = self.cluster_info_store[cluster_name]
            cluster_members.append(i)
        self.ever_random = ever_random
        self.cluster_keys = None 
        self.train_examples = self.get_reset_samples() 

    def get_reset_samples(self):
        train_examples = []
        cluster_info = copy.deepcopy(self.cluster_info_store)
        keys = list(self.cluster_info_store.keys())
        random.shuffle(keys)
        for k in keys:
            random.shuffle(cluster_info[k])
        index = 0
        while keys:
            i = index % len(keys)
            k = keys[i]
            info = cluster_info[k]
            train_examples.append(info.pop())
            if len(info) == 0:
                del keys[i]
                del cluster_info[k] 
            index +=1
        return train_examples

    def __getitem__(self, index):
        begin = index * self.batch_size
        end = begin + self.batch_size
        result = self.train_examples[begin:end]
        training_set, _ = self.data_manager.get_training_and_test_set()
        if self.debug_file:
            debug_df = training_set.iloc[result][['Entry', 'Entry name', 'EC number', 'Cluster name']]
            debug_df.to_csv(self.debug_file, sep='\t', index=False, mode='a')
        x, y = self.data_manager.get_training_data()
        rx = x[result]
        ry = []
        task_num = self.data_manager.get_task_num()
        for i in range(task_num):
            ry.append(y[i][result])
        return rx, ry

    def __len__(self):
        return self.batch_num

    def on_epoch_end(self):
        if self.ever_random:
            self.train_examples = self.get_reset_samples()
        self.debug_file = None
