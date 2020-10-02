import tensorflow as tf
import os.path
import copy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization, AveragePooling1D
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from framework import utili
from framework.model_manager import model_manager
from framework.model_manager import model_manager_creator 
from framework.model_manager import dense_net


def create(data_manager, config):
    return dense_net_manager(data_manager, config)

class dense_net_manager(model_manager.model_common_manager):
    name = 'dense_net_manager'
    def __init__(self, data_manager, config):
        super().__init__(data_manager, config)

    def _create_input(self):
        max_len = self.data_manager.get_max_len()
        max_features = self.data_manager.get_max_feature()
        embedding_dims = self.config['embedding_dims']
        inputLayer = Input(shape=(max_len,))
        return inputLayer, Embedding(max_features,
                            embedding_dims,
                            input_length=max_len)(inputLayer)

    def _create_main_path(self, lastLayer):
        dense_type = utili.get_table_value(self.config, 'dense_type', 'd121')
        dense_k = utili.get_table_value(self.config, 'dense_k', 12)
        conv_kernel_width = utili.get_table_value(self.config, 'conv_kernel_width', 3)
        bottleneck_size = utili.get_table_value(self.config, 'bottleneck_size', 1)
        transition_pool_size = utili.get_table_value(self.config, 'transition_pool_size', 2)
        transition_pool_stride = utili.get_table_value(self.config, 'transition_pool_stride', 1)
        theta = utili.get_table_value(self.config, 'theta', 1)
        initial_conv_width = utili.get_table_value(self.config, 'initial_conv_width', 3)
        initial_stride = utili.get_table_value(self.config, 'initial_stride', 1)
        initial_filters = utili.get_table_value(self.config, 'initial_filters', 48)
        initial_pool_width = utili.get_table_value(self.config, 'initial_pool_width', 2)
        initial_pool_stride = utili.get_table_value(self.config, 'initial_pool_stride', 1)
        use_global_pooling = utili.get_table_value(self.config, 'use_global_pooling', False)
        #it can be d121,d169,d201 or d264 
        if dense_type == 'd121':
            lastLayer = dense_net.DenseNet121(dense_k, conv_kernel_width, bottleneck_size, transition_pool_size, transition_pool_stride, theta, initial_conv_width, initial_stride, initial_filters, initial_pool_width, initial_pool_stride, use_global_pooling)(lastLayer)
        elif dense_type == 'd169': 
            lastLayer = dense_net.DenseNet169(dense_k, conv_kernel_width, bottleneck_size, transition_pool_size, transition_pool_stride, theta, initial_conv_width, initial_stride, initial_filters, initial_pool_width, initial_pool_stride, use_global_pooling)(lastLayer)
        elif dense_type == 'd201': 
            lastLayer = dense_net.DenseNet201(dense_k, conv_kernel_width, bottleneck_size, transition_pool_size, transition_pool_stride, theta, initial_conv_width, initial_stride, initial_filters, initial_pool_width, initial_pool_stride, use_global_pooling)(lastLayer)
        elif dense_type == 'd264': 
            lastLayer = dense_net.DenseNet264(dense_k, conv_kernel_width, bottleneck_size, transition_pool_size, transition_pool_stride, theta, initial_conv_width, initial_stride, initial_filters, initial_pool_width, initial_pool_stride, use_global_pooling)(lastLayer)

        lastLayer = Flatten()(lastLayer)
        return lastLayer
    
    def _create_end(self, input_layer, lastLayer):
        last_activation = utili.get_table_value(self.config,'last_activation', 'sigmoid')
        print('last_activation:', last_activation)
        output = []
        task_loss_num = 1
        train_target = None 
        test_target = None 
        for i in range(self.data_manager.get_task_num()):
            task_lastLayer = Dense(self.config['hidden_width'], activation='relu')(lastLayer)
            task_lastLayer = Dense(self.data_manager.get_max_category()[i], activation=last_activation, name="task_%d_1" % i)(task_lastLayer)
            output.append(task_lastLayer)
        model = Model(inputs=input_layer, outputs=output)
        return model

model_manager_creator.instance.register(dense_net_manager.name, create)
