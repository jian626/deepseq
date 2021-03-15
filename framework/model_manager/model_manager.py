import tensorflow as tf
import os.path
from datetime import datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization, AveragePooling1D
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow import math
from tensorflow import reduce_sum
from tensorflow.keras.backend import int_shape
from framework import utili
from framework.algorithm import loss_function as my_loss

class model_common_manager:
    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        if 'save_path' in config:
            save_path = config['save_path']
            dirname = os.path.dirname(save_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        self.config = config
        self.config['name'] = self.name
        self.context = {} 

    def create_model(self):
        input_layer, lastLayer = self._create_input()
        lastLayer = self._create_main_path(lastLayer)
        self.context['model'] = self._create_end(input_layer, lastLayer)
        return self.context['model']

    def compile(self, loss_weights=None): 
        task_num = self.data_manager.get_task_num()
        optimizer = self.config['optimizer']
        loss_function = self.config['loss_function']
        if loss_weights:
            def get_loss(s_weights):
                def cus_fun(y_true_and_weights, y_pred):
                    rn = 31 
                    cln = int(s_weights.shape[1])
                    y_true = tf.slice(y_true_and_weights, [0, 0], [rn, cln])
                    weights = tf.slice(y_true_and_weights, [0, cln], [rn, cln])
                    return reduce_sum(math.multiply(math.add(math.multiply(y_true, math.log(y_pred)), math.multiply(math.subtract(1.0, y_true), math.log(math.subtract(1.0, y_pred)))), weights))
                return cus_fun
            losses = []
            for weight in loss_weights:
                losses.append(get_loss(weight))
            self.get_model().compile(optimizer=optimizer, loss=losses,  metrics=['categorical_accuracy'] * task_num)
        else:
            self.get_model().compile(optimizer=optimizer, loss=[loss_function] * task_num , metrics=['categorical_accuracy'] * task_num)

    def fit_active_training(self, x_train, y_train, epochs, batch_size):
        active_training_config = None
        if 'active_training' in self.config:
            active_training_config = self.config['active_training']
        self.get_model().fit(x_train, y_train, epochs=1,  batch_size=batch_size, validation_split=1/6)
        y = []
        for i in range(4):
            y.append(y_train[i][:200])
        training_loss = self.get_model().test_on_batch(x_train[:batch_size], y)
        print(training_loss)

    def fit(self, x_train, y_train, epochs, batch_size, train_loss_weight=None, test_loss_weight=None):  
        callbacks = []
        task_num = self.data_manager.get_task_num()
        if (task_num == 1) and self.config['early_stopping']:
            patience = self.config['patience']
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True, patience=patience, verbose=1)
            callbacks.append(early_stopping_callback)
        self.get_model().fit(x_train, y_train, epochs=epochs,  batch_size=batch_size, validation_split=1/6, callbacks=callbacks)

    def fit_generator(self, generator, epochs):
        self.get_model().fit_generator(generator, epochs=epochs)

    def predict(self, x_):
        begin = datetime.now()
        current_time = begin.strftime("%H:%M:%S")
        print("***prediction begin time***:", current_time)

        res = self.model.predict(x_)

        end = datetime.now()
        current_time = end.strftime("%H:%M:%S")
        print("***end time***:", current_time)
        print("total prediction time cost:", end - begin)
        return res

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
                'config':self.config,
                'data_manager_info':self.data_manager.get_encode_info(),
                'name':self.config['name']
            }
            utili.save_obj(store, save_name)
            self.context['model'] = model

    def set_model(self, model):
        self.context['model'] = model

    def load_model(self, name):
        model_name = name + '.h5'
        model = load_model(model_name)
        self.set_model(model)

    def get_data_manager(self):
        return self.data_manager

    def predict(self, x_data):
        return self.context['model'].predict(x_data)

    def predict_on_file(self, load_file):
        data, entry_name = self.data_manager.load_x_from_file(load_file)
        return self.predict(data), entry_name
