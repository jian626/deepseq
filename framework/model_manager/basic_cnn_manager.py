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


def create(data_manager, config):
    return basic_cnn_manager(data_manager, config)

class basic_cnn_manager(model_manager.model_common_manager):
    name = 'basic_cnn_manager'
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
        pooling_strides = self.config['pooling_strides']
        kernelSize = self.config['conv_kernel_width']
        pool_size = self.config['pool_size']
        layer_len = self.config['layer_len']
        cov_len = self.config['cov_len']
        
        kernelSize = self.config['conv_kernel_width']
        delta = self.config['filter_delta']
        pool_interval = utili.get_table_value(self.config, 'pool_inverval', 2)
        drop_out_rate = utili.get_table_value(self.config, 'drop_out_rate', 0.2)
        start_width = utili.get_table_value(self.config, 'start_width', 48)
        for i in range(layer_len): 
            for j in range(cov_len):
                lastLayer = Conv1D(start_width +delta * j, kernelSize, padding='same', activation='relu')(lastLayer)
                if j % pool_interval == 0:
                    lastLayer = Dropout(drop_out_rate)(lastLayer)
                    lastLayer = MaxPooling1D(pool_size=pool_size, strides=pooling_strides, padding='same')(lastLayer)
        lastLayer = Flatten()(lastLayer)
        lastLayer = Dropout(drop_out_rate)(lastLayer)
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


model_manager_creator.instance.register(basic_cnn_manager.name, create)
