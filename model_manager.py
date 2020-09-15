import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization, AveragePooling1D
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import utili

class model_creator:
    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        self.config = config
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def create_input(self):
        max_len = self.data_manager.get_max_len()
        max_features = self.data_manager.get_max_feature()
        embedding_dims = self.config['embedding_dims']
        inputLayer = Input(shape=(max_len,))
        return inputLayer, Embedding(max_features,
                            embedding_dims,
                            input_length=max_len)(inputLayer)

    def create_main_path(self, input_layer):
        lastLayer = input_layer
        dense_net = self.config['dense_net']
        pooling_strides = self.config['pooling_strides']
        kernelSize = self.config['cov_kernel_size']
        pool_size = self.config['pool_size']
        layer_len = self.config['layer_len']
        cov_len = self.config['cov_len']
        
        if dense_net:
            lastLayer_1 = lastLayer
            lastLayer_2 = lastLayer
            lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Dropout(0.2)(lastLayer_1)
            lastLayer_1 = AveragePooling1D(pool_size=pool_size, strides=pooling_strides, padding='same')(lastLayer_1)
            lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Dropout(0.2)(lastLayer_1)
            lastLayer_1 = AveragePooling1D(pool_size=pool_size, strides=pooling_strides, padding='same')(lastLayer_1)
            lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Dropout(0.2)(lastLayer_1)
            lastLayer_1 = AveragePooling1D(pool_size=pool_size, strides=pooling_strides, padding='same')(lastLayer_1)
            lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Conv1D(16, kernelSize, padding='same', activation='relu')(lastLayer_1)
            lastLayer_1 = Dropout(0.2)(lastLayer_1)
            lastLayer_1 = MaxPooling1D(pool_size=pool_size, strides=pooling_strides, padding='same')(lastLayer_1)
            mainLayer = tf.keras.layers.Add()([lastLayer_1, lastLayer_2])
            lastLayer_3 = mainLayer
            mainLayer = Conv1D(48, kernelSize, padding='same', activation='relu')(mainLayer)
            mainLayer = Conv1D(48, kernelSize, padding='same', activation='relu')(mainLayer)
            mainLayer = Conv1D(64, kernelSize, padding='same', activation='relu')(mainLayer)
            mainLayer = Conv1D(64, kernelSize, padding='same', activation='relu')(mainLayer)
            mainLayer = MaxPooling1D(pool_size=pool_size, strides=pooling_strides, padding='same')(mainLayer)
            mainLayer = Dropout(0.2)(mainLayer)
            mainLayer = Conv1D(48, kernelSize, padding='same', activation='relu')(mainLayer)
            mainLayer = Conv1D(48, kernelSize, padding='same', activation='relu')(mainLayer)
            mainLayer = Conv1D(64, kernelSize, padding='same', activation='relu')(mainLayer)
            mainLayer = Conv1D(64, kernelSize, padding='same', activation='relu')(mainLayer)
            mainLayer = MaxPooling1D(pool_size=pool_size, strides=pooling_strides, padding='same')(mainLayer)
            mainLayer = Dropout(0.2)(mainLayer)
            lastLayer = tf.keras.layers.Concatenate()([mainLayer, lastLayer_3])
            
            lastLayer = Dropout(0.2)(lastLayer)
            lastLayer = Flatten()(lastLayer)
            lastLayer = Dense(256)(lastLayer)
            lastLayer = Dropout(0.2)(lastLayer)
        else:
            kernelSize = self.config['cov_kernel_size']
            delta = self.config['filter_delta']
            for i in range(layer_len): 
                for j in range(cov_len):
                    lastLayer = Conv1D(48+delta * j, kernelSize, padding='same', activation='relu')(lastLayer)
                    if j % 2 == 0:
                        lastLayer = MaxPooling1D(pool_size=pool_size, strides=pooling_strides, padding='same')(lastLayer)
            lastLayer = Flatten()(lastLayer)
        return lastLayer
    
    def create_end(self, input_layer, lastLayer):
        output = []
        task_loss_num = 1
        train_target = None 
        test_target = None 
        for i in range(self.data_manager.get_task_num()):
            task_lastLayer = Dense(self.config['hidden2Dim'])(lastLayer)
            task_lastLayer = Dense(self.data_manager.get_max_category()[i], activation='sigmoid', name="task_%d_1" % i)(task_lastLayer)
            output.append(task_lastLayer)
        model = Model(inputs=input_layer, outputs=output)
        return model
        

    def create_model(self):
        input_embedding_layer, lastLayer = self.create_input()
        lastLayer = self.create_main_path(lastLayer)
        self.model = self.create_end(input_embedding_layer, lastLayer)
        return self.model
    
    def get_model(self):
        return self.model

    def save_model(self, suffix):
        save_model_name = utili.get_table_value(self.config, 'save_model_name')
        if save_model_name: 
            save_model_name + suffix 

            save_path = utili.get_table_value(self.config, 'save_path', './')
            save_name = save_path + '/' + save_model_name 
            self.model.save_weights(save_name+ '.h5')
            config = {
                'specific':self.data_manager.get_specific_info(),
                'task_num':self.data_manager.get_task_num(),
                'field_map_to_number':self.data_manager.get_feature_mapping(),
            }
            utili.save_obj(config, save_name)
