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
from datetime import datetime
    
class estimator_manager:
    def __init__(self, config, data_manager, model_manager, estimators):
        self.config = config
        self.data_manager = data_manager 
        self.model_manager = model_manager 
        self.estimators = estimators

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
        
        y_train = y_train
        batch_size = self.config['batch_size']
            
        if not cur_round is None:
            print('***************current runing is based on %d round, this run will has %d epochs.****************' % (cur_round, epochs))

        model.fit(x_train, y_train, epochs=epochs,  batch_size=batch_size, validation_split=1/6, callbacks=callbacks)

        suffix = ''
        if not cur_round is None:
            suffix += '_round_' + str(cur_round)

        self.model_manager.save_model(suffix)
            
        y_pred = model.predict(x_test)

        for estimator in self.estimators:
            estimator.estimate(y_pred, y_test, len(x_test), self.config['print_report'])
        return 
