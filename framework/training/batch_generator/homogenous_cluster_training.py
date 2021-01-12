from framework.training.batch_generator.training_base import cluster_training_base
from framework.training.batch_generator import batch_generator_creator as creator
import numpy as np
import random
import copy
class homogenous_cluster_training(cluster_training_base):
    name = 'homogenous_cluster_training'
    def reset_samples(self):
        train_examples = []
        keys = list(self.cluster_info_store.keys())
        random.shuffle(keys)
        for k in keys:
            random.shuffle(self.cluster_info_store[k])
            train_examples += self.cluster_info_store[k]
        self.train_examples = train_examples

def create(config, data_manager):
    return homogenous_cluster_training(config, data_manager)

creator.instance.register(homogenous_cluster_training.name, create)
