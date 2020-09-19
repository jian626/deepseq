import numpy as np
import pandas as pd
from framework import utili
from framework.bio import process_enzyme
from framework.bio import BioDefine
from tensorflow.keras.preprocessing import sequence

class enzyme_data_manager:
    name = 'enzyme_data_manager'
    def __init__(self, config):
        self.config = config
        self.config['name'] = 'enzyme_data_manager'
        self.label_key = config['label_key']
        if not 'class_maps' in self.config:
            self.config['class_maps'] = { 
                    0:{},
                    1:{},
                    2:{},
                    3:{}
               } 
    
        if not 'field_map_to_number' in self.config:
            self.config['field_map_to_number'] = {
                    0:{},
                    1:{},
                    2:{},
                    3:{}
                }

        if not 'number_to_field' in self.config:
            self.config['number_to_field'] = {
                }

    def ___apply_threshold(self, df, level, threshold):
        temp = {}
        def count_class_example(ec_list):
            for ec in ec_list: 
                ec = process_enzyme.get_ec_to_level(ec, level)
                if ec in temp:
                    temp[ec] += 1
                else:
                    temp[ec] = 1
            
        
        def delete_class(ec_list, threshold):
            res = []
            for ec in ec_list:
                ec = process_enzyme.get_ec_to_level(ec, level)
                if temp[ec] >  threshold:
                    res.append(ec)
            return res

        df[self.label_key].apply(lambda e:count_class_example(e))  

        return df[self.label_key].apply(lambda e:delete_class(e, threshold))


    def _apply_threshold(self, df):
        class_example_threshhold = self.config['class_example_threshhold']
        size = df.shape[0]
        ec_level = self.config['ec_level']
        while True:
            for i in range(ec_level-1, -1, -1):
                df[self.label_key] = self.___apply_threshold(df, i, class_example_threshhold)
            df = df[df[self.label_key].apply(lambda e:len(e)>0)]

            if size == df.shape[0]: 
                break
            else:
                size = df.shape[0]
        return df


    def get_data(self, sep=','):
        df = pd.read_csv(self.config['file_path'],sep=sep)
        utili.print_debug_info(df, info=True)
        df = df.dropna()
        
        df[self.label_key] = df[self.label_key].astype(str)
        utili.print_debug_info(df, 'after drop na', print_head = True)
        ec_level = self.config['ec_level']
        
        if self.config['drop_multilabel']:
            df = df[df[self.label_key].apply(lambda x:process_enzyme.not_multilabel_enzyme(x))]

        if not self.config['apply_dummy_label']:
            df[self.label_key]= df[self.label_key].apply(lambda x:process_enzyme.get_ec_level_list(x, ec_level))
            df = df[df[self.label_key].apply(lambda x:len(x)>0)]
        else:
            df[self.label_key]= df[self.label_key].apply(lambda x:process_enzyme.get_ec_list(x))
            
        df['EC count'] = df[self.label_key].apply(lambda x:len(x))

        if self.config['max_len'] > 0:
            df = df[df['Sequence'].apply(lambda x:len(x)<=self.config['max_len'])]
            utili.print_debug_info(df, 'after drop seq more than %d ' % self.config['max_len'], print_head = True)

        df = self._apply_threshold(df)

        utili.print_debug_info(df, 'after apply threshold', print_head = True)
        
        for i in range(ec_level):
            df = process_enzyme.get_level_labels(df, i, self.config['class_maps'])
            utili.print_debug_info(df, 'after select to level %d' % i, print_head = True)
        
        self.config['max_category'] = []
        for i in range(ec_level):
            df, temp_max_category, temp_field_map_to_number = process_enzyme.create_label_from_field(df, self.config['class_maps'],'level%d' % i, 'task%d' % i, i)
            self.config['max_category'].append(temp_max_category)
            self.config['field_map_to_number'][i] = temp_field_map_to_number
            utili.print_debug_info(df, 'after create task label to level %d' % i, print_head = True)
        print('max_category:', self.config['max_category'])
        
        if self.config['print_statistics']:
            print('following statistics information is based on data to use.')
            for index in range(ec_level):
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

        task_num = self.get_task_num() 
        if task_num == 1:
            y_train = [y_train[ec_level - 1]]
            y_test = [y_test[ec_level - 1]]

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train, x_test, y_test

    def get_training_data(self):
        return self.x_train, self.y_train
    
    def get_test_data(self):
        return self.x_test, self.y_test

    def get_task_num(self):
        return self.config['task_num']

    def get_max_category(self):
        ret = self.config['max_category']
        if self.get_task_num() == 1:
            ret = [self.config['max_category'][self.config['ec_level']-1]]
        return ret

    def get_max_feature(self):
        return self.config['max_features']

    def get_max_len(self):
        return self.config['max_len']

    def _one_hot_to_labels(self, labels, task_id, map_table):
        category_num = self.config['max_category'][task_id]
        res = []
        for label in labels:
            temp = []
            for index, element in enumerate(label): 
                if element:
                    temp.append(map_table[index])
            if temp:
                temp = ';'.join(temp)
            else:
                temp = 'uncertain'
            res.append(temp)
        return res

    def one_hot_to_labels(self, y):
        task_num = self.get_task_num()
        number_to_field = self.config['number_to_field']
        ret = []
        if task_num == 4: 
            for i in range(task_num):
                if not i in number_to_field:
                    number_to_field[i] = utili.switch_key_value(self.config['field_map_to_number'][i])
                labels = self._one_hot_to_labels(y[i], i, number_to_field[i])
                ret.append(labels)
        else:
            ec_level = self.config['ec_level']
            i = ec_level - 1
            if not i in number_to_field:
                number_to_field[i] = utili.switch_key_value(self.config['field_map_to_number'][i])
            ret.append(self._one_hot_to_labels(y[0], i, number_to_field[i]))
        return ret

    def get_encode_info(self):
        return self.config

    def get_class_statistic(self, c):
        class_maps = self.config['class_maps']
        for k in class_maps:
            if c in class_maps[k]:
                cnt = class_maps[k][c]
                return cnt, k+1
                 
    def get_x(self, df):
        max_len = self.config['max_len']
        def check_len(seq):
            if len(seq) > max_len:
                raise Exception('len %d beyone max_len:%s' % (len(seq), seq))
        df = df[df['Sequence'].apply(lambda x:len(x)<max_len)]
        df['Sequence'].apply(check_len)
        feature_list = utili.GetNGrams(BioDefine.aaList, self.config['ngram'])
        x = df['Sequence'].apply(lambda x:utili.GetOridinalEncoding(x, feature_list, self.config['ngram']))
        return sequence.pad_sequences(x, maxlen=max_len, padding='post'), df['Entry name']

    def load_x_from_file(self, file_name):
        df = pd.read_csv(file_name, sep='\t')
        return self.get_x(df) 
