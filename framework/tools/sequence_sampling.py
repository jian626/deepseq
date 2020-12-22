from tensorflow.keras.utils import Sequence
import numpy as np
import random
import copy
class SequenceGenerator(Sequence):
    def __init__(self, data_manager, batch_size):
        self.data_manager = data_manager
        self.batch_size = batch_size
        x, y = self.data_manager.get_training_data()
        self.sample_len = None 
        self.cluster_info = None
        self.sample_len_store = len(x)
        print('sample_len:', self.sample_len)
        self.batch_num = int(np.floor(len(x) / self.batch_size))
        self.cluster_info_store = {} 
        training_set, _ = self.data_manager.get_training_and_test_set()
        print('Cluster name')
        for i in range(training_set.shape[0]):
            cluster_name = training_set.iloc[i]['Cluster name']
            cluster_members = None
            if not cluster_name in self.cluster_info_store:
                self.cluster_info_store[cluster_name] = []
            cluster_members = self.cluster_info_store[cluster_name]
            cluster_members.append(i)

        self.reset()

    def reset(self):
        self.sample_len = self.sample_len_store
        self.cluster_info = copy.deepcopy(self.cluster_info_store)

    def __getitem__(self, index):
        x, y = self.data_manager.get_training_data()
        task_num = self.data_manager.get_task_num()
        cluster_keys = list(self.cluster_info.keys())
        result = []
        sample_len = self.sample_len
        needed = min(self.batch_size, sample_len)

        print(sample_len, needed, index)

        if len(cluster_keys) >= needed:
            for i in range(needed):
                key = cluster_keys[(i+(index * self.batch_size))% len(cluster_keys)]
                members = self.cluster_info[key]
                result.append(members.pop()) 
                sample_len -= 1
                if not members:
                    del self.cluster_info[key]
        else:
            while needed>len(result):
                cluster_keys = list(self.cluster_info.keys())
                for k in cluster_keys:
                    members = self.cluster_info[k]
                    result.append(members.pop())
                    if not members:
                        del self.cluster_info[k]
                    sample_len -= 1
                    if needed <= len(result):
                        break

        self.sample_len = sample_len
        training_set, _ = self.data_manager.get_training_and_test_set()
        rx = x[result] 
        ry = []

        for i in range(task_num):
            ry.append(y[i][result])

        return rx, ry



    def __len__(self):
        return self.batch_num

    def on_epoch_end(self):
        self.reset()
