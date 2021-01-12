from framework.training.batch_generator.training_base import cluster_training_base
import numpy as np
import random
import copy
class homogenous_cluster_training(cluster_training_base):
    def reset_samples(self):
        for _, v in self.cluster_info_store.items():
            train_examples += v

        self.train_examples = train_examples
