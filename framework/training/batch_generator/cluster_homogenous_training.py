from framework.training.batch_generator.training_base import Sequence
import numpy as np
import random
import copy
class homogenous_cluster_training(cluster_training_base):
    def reset_samples(self):
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
        self.train_examples = train_examples
