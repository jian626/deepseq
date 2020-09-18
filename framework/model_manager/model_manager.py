import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization, AveragePooling1D
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from framework import utili

class model_creator:
    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        self.config = config
        self.context = {} 

    def _create_input(self):
        max_len = self.data_manager.get_max_len()
        max_features = self.data_manager.get_max_feature()
        embedding_dims = self.config['embedding_dims']
        inputLayer = Input(shape=(max_len,))
        return inputLayer, Embedding(max_features,
                            embedding_dims,
                            input_length=max_len)(inputLayer)

    def _create_main_path(self, input_layer):
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
    
    def _create_end(self, input_layer, lastLayer):
        last_activation = utili.get_table_value(self.config,'last_activation', 'sigmoid')
        output = []
        task_loss_num = 1
        train_target = None 
        test_target = None 
        for i in range(self.data_manager.get_task_num()):
            task_lastLayer = Dense(self.config['hidden_width'])(lastLayer)
            task_lastLayer = Dense(self.data_manager.get_max_category()[i], activation=last_activation, name="task_%d_1" % i)(task_lastLayer)
            output.append(task_lastLayer)
        model = Model(inputs=input_layer, outputs=output)
        return model

    def create_model(self):
        input_embedding_layer, lastLayer = self._create_input()
        lastLayer = self._create_main_path(lastLayer)
        self.context['model'] = self._create_end(input_embedding_layer, lastLayer)
        return self.context['model']

    def compile(self): 
        task_num = self.data_manager.get_task_num()
        optimizer = self.config['optimizer']
        loss_function = self.config['loss_function']
        self.get_model().compile(optimizer=optimizer, loss=[loss_function] * task_num , metrics=['categorical_accuracy'] * task_num)

    def fit(self, x_train, y_train, epochs, batch_size):  
        callbacks = []
        task_num = self.data_manager.get_task_num()
        if (task_num == 1) and self.config['early_stopping']:
            patience = self.config['patience']
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True, patience=40, verbose=1)
            callbacks.append(early_stopping_callback)
        self.get_model().fit(x_train, y_train, epochs=epochs,  batch_size=batch_size, validation_split=1/6, callbacks=callbacks)

    def predict(self, x_):
        return self.model.predict(x_)
        

    def get_summary(self):
        return self.get_model().summary()
    
    def get_model(self):
        return self.context['model']

    def save_model(self, suffix=""):
        save_model_name = utili.get_table_value(self.config, 'save_model_name')
        if save_model_name: 
            save_model_name + suffix 

            save_path = utili.get_table_value(self.config, 'save_path', './')
            save_name = save_path + '/' + save_model_name 
            self.context['model'].save(save_name+ '.h5')
            model = self.context['model']
            self.context['model'] = None
            store = {
                'obj':self
            }
            utili.save_obj(store, save_name)
            self.context['model'] = model

    def set_model(self, model):
        self.context['model'] = model

    def load_model(name):
        model_name = name + '.h5'
        model = load_model(model_name)
        store = utili.load_obj(name)
        model_creator = store['obj']
        model_creator.set_model(model)
        return model_creator

    def get_data_manager(self):
        return self.data_manager

    def predict(self, x_data):
        return self.context['model'].predict(x_data)

    def predict_on_file(self, load_file):
        data = self.data_manager.load_x_from_file(load_file)
        return self.predict(data)
