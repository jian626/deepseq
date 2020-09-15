import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization, AveragePooling1D
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import utili
import BioDefine
from datetime import datetime
from sklearn.metrics import classification_report
import process_enzyme
    
class model_estimator:
    def __init__(self, config, data_manager, model_manager):
        self.config = config
        self.data_manager = data_manager 
        self.model_manager = model_manager 

    def evaluate(self):
        epochs = self.config['epochs']
        batch_round = utili.get_table_value(self.config, 'batch_round')

        model = self.model_manager.get_model()
        x_train, y_train = self.data_manager.get_training_data()
        x_test, y_test = self.data_manager.get_test_data()

        task_num = self.data_manager.get_task_num()

        optimizer = self.config['optimizer']
        model.compile(optimizer=optimizer, loss=['binary_crossentropy'] * task_num , metrics=['categorical_accuracy'] * task_num)

        if self.config['print_summary']:
            print(model.summary())

        if batch_round:
            round_size = utili.get_table_value(self.config, 'round_size', 10)
            total_size = (epochs + round_size - 1) // round_size
            for i in range(total_size):
                self._evaluate(model, x_train, y_train, x_test, y_test, round_size, i)
        else:
            self._evaluate(model, x_train, y_train, x_test, y_test, epochs)

    def _evaluate(self, model, x_train, y_train, x_test, y_test, epochs, cur_round = None):

        task_num = self.data_manager.get_task_num()
        
        callbacks = []
        if (task_num == 1) and self.config['early_stopping']:
            patience = self.config['patience']
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True, patience=40, verbose=1)
            callbacks.append(early_stopping_callback)
        
        y_train_target = y_train
        y_test_target = y_test
        batch_size = self.config['batch_size']
            
        if not cur_round is None:
            print('***************current runing is based on %d round, this run will has %d epochs.****************' % (cur_round, epochs))

        model.fit(x_train, y_train_target, epochs=epochs,  batch_size=batch_size, validation_split=1/6, callbacks=callbacks)

        suffix = ''
        if not cur_round is None:
            suffix += '_round_' + str(cur_round)

        self.model_manager.save_model(suffix)
            
        y_pred = model.predict(x_test)

        if task_num == 1:
            y_pred = [y_pred]

        field_map_to_number = self.data_manager.get_feature_mapping()
        map_table = {} 
        class_res = {
        }
        

        for i in range(task_num):
            pred = (y_pred[i] > 0.5)
            target = y_test_target[i]
            report = classification_report(target, pred)
            map_table[i] = utili.switch_key_value(field_map_to_number[i])

            if self.config['print_report']:
                print('report level %d' % i)
                print(report)
                res = utili.strict_compare_report(target, pred, len(x_test))
                print('strict accuracy is %d of %d, %f%%' % (res, len(x_test), float(res) * 100.0 / len(x_test)))

            temp = []
            for y_ in pred:
                temp.append(utili.map_label_to_class(map_table[i], y_))
            class_res[i] = temp

            res = {
                0:0, 
                1:0,
                2:0,
            }

        for i in range(len(x_test)):
            for j in range(task_num-1):
                if process_enzyme.is_conflict(class_res[j+1][i], class_res[j][i], j+1):
                    res[j] += 1

        for i in range(task_num-1):
            print('comflict between level %d and level %d is %d, %f%% of %d.' % (i+1, i+2, res[i], float(res[i]) * 100.0 /float(len(x_test)), len(x_test)))
                
        return report
