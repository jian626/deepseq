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

class data_processor:
    def __init__(self, config):
        self.config = config
        self.config['class_maps'] = { 
                0:{},
                1:{},
                2:{},
                3:{}
           } 

        self.config['field_map_to_number'] = {
                0:{},
                1:{},
                2:{},
                3:{}
            }

    def get_data(self, sep=','):
        df = pd.read_csv(self.config['file_path'],sep=sep)
        utili.print_debug_info(df, info=True)
        
        df = df.dropna()
        
        df['EC number'] = df['EC number'].astype(str)
        utili.print_debug_info(df, 'after drop na', print_head = True)
        
        if self.config['drop_multilabel']:
            df = df[df['EC number'].apply(lambda x:process_enzyme.not_multilabel_enzyme(x))]
            utili.print_debug_info(df, 'after drop multilabel')
            if not self.config['apply_dummy_label']:
                utili.print_debug_info(df, 'before drop dummy')
                df = df[df['EC number'].apply(lambda x:process_enzyme.has_level(ec_level, x))]
                utili.print_debug_info(df, 'after drop dummy')
        else:
            if not self.config['apply_dummy_label']:
                df['EC number']= df['EC number'].apply(lambda x:process_enzyme.get_ec_level_list(x, self.config['ec_level']))
                df = df[df['EC number'].apply(lambda x:len(x)>0)]
            else:
                df['EC number']= df['EC number'].apply(lambda x:process_enzyme.get_ec_list(x))
                
            df['EC count'] = df['EC number'].apply(lambda x:len(x))
        
            
        
        if self.config['max_len'] > 0:
            df = df[df['Sequence'].apply(lambda x:len(x)<=self.config['max_len'])]
            utili.print_debug_info(df, 'after drop seq more than %d ' % self.config['max_len'], print_head = True)
        
        for i in range(self.config['ec_level']):
            df = process_enzyme.get_level_labels(df, i, self.config['class_maps'])
            utili.print_debug_info(df, 'after select to level %d' % i, print_head = True)
        
        
        
        self.config['max_category'] = []
        for i in range(self.config['ec_level']):
            df, temp_max_category, temp_field_map_to_number = process_enzyme.create_label_from_field(df, self.config['class_maps'],'level%d' % i, 'task%d' % i, i)
            self.config['max_category'].append(temp_max_category)
            self.config['field_map_to_number'][i] = temp_field_map_to_number
            utili.print_debug_info(df, 'after create task label to level %d' % i, print_head = True)
        print('max_category:', self.config['max_category'])
        
        if self.config['print_statistics']:
            print('following statistics information is based on data to use.')
            for index in range(self.config['ec_level']):
                sorted_k = {k: v for k, v in sorted(self.config['class_maps'][index].items(), key=lambda item: item[1])}
                cnt = 0
                map_cnt = {}
                for k in sorted_k: 
                    if not sorted_k[k] in map_cnt:
                        map_cnt[sorted_k[k]] = 1
                    else:
                        map_cnt[sorted_k[k]] += 1
        
                less_than_10 = 0
                for i in range(10):
                    if i in map_cnt:
                        less_than_10 += map_cnt[i]
                print('level %d: %d classes less than 10, occupy %f%% of %d' % (index+1, less_than_10, float(less_than_10) * 100.0 / self.config['max_category'][index], self.config['max_category'][index]))
        
        
        df = df.sample(frac=self.config['fraction'])
        utili.print_debug_info(df, 'after sampling frac=%f' % self.config['fraction'])
        self.config['using_set_num'] = df.shape[0]
        df = df.reindex(np.random.permutation(df.index))
        
        self.config['max_len'] = 0
        def set_max_len(x):
            if self.config['max_len'] <len(x):
                self.config['max_len'] = len(x)
            
        df['Sequence'].apply(lambda x:set_max_len(x))
        print('max_len:', self.config['max_len'])
        feature_list = utili.GetNGrams(BioDefine.aaList, self.config['ngram'])
        self.config['max_features'] = len(feature_list) + 1
        df['Encode'] = df['Sequence'].apply(lambda x:utili.GetOridinalEncoding(x, feature_list, self.config['ngram']))
        
        
        print('train_percent:%f' % self.config['train_percent'])
        training_set = df.iloc[:int(self.config['using_set_num'] * self.config['train_percent'])]
        test_set = df.iloc[training_set.shape[0]:]
        utili.print_debug_info(training_set, "training set", print_head=True)
        utili.print_debug_info(test_set, "test set", print_head=True)
        
        x_train, y_train = process_enzyme.get_data_and_label(training_set, self.config)
        x_test, y_test = process_enzyme.get_data_and_label(test_set, self.config)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train, x_test, y_test

    def get_training_data(self):
        return self.x_train, self.y_train
    
    def get_test_data(self):
        return self.x_test, self.y_test
    
class model_creator:
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def create_input(self):
        max_len = self.data_config['max_len']
        max_features = self.data_config['max_features']
        embedding_dims = self.model_config['embedding_dims']
        inputLayer = Input(shape=(max_len,))
        return inputLayer, Embedding(max_features,
                            embedding_dims,
                            input_length=max_len)(inputLayer)

    def create_main_path(self, input_layer):
        lastLayer = input_layer
        dense_net = self.model_config['dense_net']
        pooling_strides = self.model_config['pooling_strides']
        kernelSize = self.model_config['cov_kernel_size']
        pool_size = self.model_config['pool_size']
        layer_len = self.model_config['layer_len']
        cov_len = self.model_config['cov_len']
        
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
            kernelSize = self.model_config['cov_kernel_size']
            delta = self.model_config['filter_delta']
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
        if self.model_config['multi_task']:
            for i in range(self.data_config['ec_level']):
                task_lastLayer = Dense(self.model_config['hidden2Dim'])(lastLayer)
                task_lastLayer = Dense(self.data_config['max_category'][i], activation='sigmoid', name="task_%d_1" % i)(task_lastLayer)
                output.append(task_lastLayer)
        else:
            max_category = self.data_config['max_category']
            ec_level = self.data_config['ec_level']
            lastLayer = Dense(max_category[ec_level-1], activation='softmax')(lastLayer)
            output.append(lastLayer)
        model = Model(inputs=input_layer, outputs=output)
        return model
        

    def create_model(self):
        input_embedding_layer, lastLayer = self.create_input()
        lastLayer = self.create_main_path(lastLayer)
        self.model = self.create_end(input_embedding_layer, lastLayer)
        return self.model
    
    def get_model(self):
        return self.model

class model_estimator:
    def __init__(self, config, data_processor, model_creator):
        self.config = config
        self.data_processor = data_processor 
        self.model_creator = model_creator 
    def evaluate(self):
        epochs = self.config['epochs']
        batch_round = utili.get_table_value(self.config, 'batch_round')

        model = self.model_creator.get_model()
        x_train, y_train = self.data_processor.get_training_data()
        x_test, y_test = self.data_processor.get_test_data()

        multi_task = self.model_creator.model_config['multi_task'] 
        task_loss_num = 1
        ec_level = self.data_processor.config['ec_level']
        if multi_task:
            task_loss_num = ec_level

        optimizer = self.config['optimizer']
        model.compile(optimizer=optimizer, loss=['binary_crossentropy'] * task_loss_num, metrics=['categorical_accuracy'])

        if self.config['print_summary']:
            print(model.summary())
        if batch_round:
            round_size = utili.get_table_value(self.config, 'round_size', 10)
            total_size = (epochs + round_size - 1) // round_size
            for i in range(total_size):
                self._evaluate(model, ec_level, x_train, y_train, x_test, y_test, round_size, i)
        else:
            self._evaluate(model, ec_level, x_train, y_train, x_test, y_test, epochs)

    def _evaluate(self, model, ec_level, x_train, y_train, x_test, y_test, epochs, cur_round = None):

        multi_task = self.model_creator.model_config['multi_task'] 
        
        callbacks = []
        if not multi_task and self.config['early_stopping']:
            patience = self.config['patience']
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True, patience=40, verbose=1)
            callbacks.append(early_stopping_callback)
        
        y_train_target = None
        if multi_task:
            y_train_target = y_train
            y_test_target = y_test
        else:
            y_train_target = y_train[ec_level-1]
            y_test_target = y_test[ec_level-1]
        batch_size = self.config['batch_size']
            
        if not cur_round is None:
            print('***************current runing is based on %d round, this run will has %d epochs.****************' % (cur_round, epochs))

        model.fit(x_train, y_train_target, epochs=epochs,  batch_size=batch_size, validation_split=1/6, callbacks=callbacks)
        if 'save_model_name' in self.model_creator.model_config:
            save_model_name = self.model_creator.model_config['save_model_name']
            if not cur_round is None:
                save_model_name + '_r_' + str(cur_round)

            save_path = './'
            if 'save_path' in self.model_creator.model_config:
                save_path = self.model_creator.model_config['save_path'] 
                save_name = save_path + '/' + save_model_name 
            model.save_weights(save_name+ '.h5')
            config = {
                'ec_level':ec_level,
                'multi_task':multi_task,
                'field_map_to_number':self.data_processor.config['field_map_to_number'],
            }
            utili.save_obj(config, save_name)
            
        y_pred = model.predict(x_test)
        if multi_task:
            field_map_to_number = self.data_processor.config['field_map_to_number']
            map_table = {} 
            class_res = {
            }
            

            for i in range(ec_level):
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
                for j in range(ec_level-1):
                    if process_enzyme.is_conflict(class_res[j+1][i], class_res[j][i], j+1):
                        res[j] += 1
            for i in range(ec_level-1):
                print('comflict between level %d and level %d is %d, %f%% of %d.' % (i+1, i+2, res[i], float(res[i]) * 100.0 /float(len(x_test)), len(x_test)))
                    
        else:
            y_pred = (y_pred > 0.5)
            report = classification_report(y_test_target, y_pred)
            if self.config['print_report']:
                print('report level %d' % ec_level)
                print(report)
                res = utili.strict_compare_report(y_test_target, y_pred, len(x_test))
                print('strict accuracy is %d% of %d, %f%%' % (res, len(x_test), float(res)*1000/len(x_test)))
        return report
