import numpy as np
import pandas as pd
from framework import utili
from framework.strategy import hierarchical_learning
from framework.bio import BioDefine
from tensorflow.keras.preprocessing import sequence
from framework.data_manager import data_manager_creator
from framework.tools import data_spliter

class enzyme_data_manager:
    name = 'new_enzyme_data_manager'
    def __init__(self, config):
        self.config = config
        self.config['name'] = 'enzyme_data_manager'
        self.label_key = config['label_key']
        self.id_name = config['id_name']
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
        self.training_set = None
        self.test_set = None

    def count_class_example(self, df, map_table, level):
        def apply_fun(label_list):
            for label in label_list: 
                label = hierarchical_learning.get_label_to_level(label, level)
                if label:
                    if label in map_table:
                        map_table[label] += 1
                    else:
                        map_table[label] = 1
        df[self.label_key].apply(lambda e:apply_fun(e))

    def ___apply_threshold(self, df, level, threshold):
        map_table = {}
            
        def delete_class_accord_threshold(label_list, threshold, map_tabel):
            res = []
            for label in label_list:
                temp = hierarchical_learning.get_label_to_level(label, level)
                if map_table[temp] >  threshold:
                    res.append(label)
            return res

        self.count_class_example(df, map_table, level)

        return df[self.label_key].apply(lambda e:delete_class_accord_threshold(e, threshold, map_table))


    def _apply_threshold(self, df):
        '''This function is used to eliminate classes with examples less than threshold
        '''
        class_example_threshhold = self.config['class_example_threshhold']
        size = df.shape[0]
        level_num = self.config['level_num']
        while True:
            for i in range(level_num-1, -1, -1):
                df[self.label_key] = self.___apply_threshold(df, i, class_example_threshhold)
            df = df[df[self.label_key].apply(lambda e:len(e)>0)]

            if size == df.shape[0]: 
                break
            else:
                size = df.shape[0]
        return df

    def map_true_value_to_number(true_values, map_table):
        ret = None
        if type(true_values) == list:
            ret = []
            for t in true_values:
                ret.append(map_table[t])
        else:
            ret = map_table[true_values]
        return ret

    def create_number_to_catogry_mapping(m):
        ret = {}
        for v, k in enumerate(m):
            ret[k] = v
        return ret

    def create_label_from_field(df, class_maps, field_name, label_name, index): 
        values = list(class_maps[index].keys())
        field_map_to_number = enzyme_data_manager.create_number_to_catogry_mapping(values)
        df[label_name] = df[field_name].apply(lambda x:enzyme_data_manager.map_true_value_to_number(x, field_map_to_number))
        return df,len(field_map_to_number), field_map_to_number

    def get_level_labels(self, df, level, class_maps):
        df["level%d" % level] = df[self.label_key].apply(lambda x:hierarchical_learning.get_label_to_level(x, level, class_maps=class_maps))
        return df

    def map_label_set_to_one_hot(self, label_set, num_classes):
        ret = None
        ret = np.zeros((len(label_set), num_classes)) 
        for index, ec in enumerate(label_set):
            for e in ec:
                ret[index][e] = 1
        return ret

    def get_x_from_df(self, df):
        x = df['Encode']
        x = sequence.pad_sequences(x, maxlen=self.config['max_len'], padding='post')
        return x

    def get_y_from_df(self, df): 
        y = []
        for i in range(self.config['level_num']):
            temp = self.map_label_set_to_one_hot(df['level%d' % i], self.config['max_category'][i])
            y.append(temp)
        return y


    def reuse_data_process(self, sep='\t'):
        reuse_data = self.config['reuse_data']
        training_set = pd.DataFrame()
        test_set = pd.DataFrame()

        for train in reuse_data['train']:
            training_set = training_set.append(pd.read_csv(train, sep=sep))

        for test in reuse_data['test']:
            test_set = test_set.append(pd.read_csv(test, sep=sep))

        print('***********training set********************')
        print(training_set.head(10))
        print('***********test set********************')
        print(test_set.head(10))

        if 'meta' in reuse_data:
            config = utili.load_obj(reuse_data['meta'])
            self.config['class_maps'] = config['class_maps']
            self.config['field_map_to_number'] = config['field_map_to_number']
            self.config['number_to_field'] = config['number_to_field']
            self.config['max_category'] = config['max_category']
            self.config['using_set_num'] = config['using_set_num']
            self.config['max_len'] = config['max_len']
            self.config['level_num'] = config['level_num']

            def convert_str_to_list(s):
                s = s[1:-1].split(',')
                ret = []
                for e in s:
                    ret.append(int(e.strip()))
                return ret 

            level = self.config['level_num']

            for i in range(level):
                training_set['level%d' % i] = training_set['level%d' % i].apply(convert_str_to_list)
                test_set['level%d' % i] = test_set['level%d' % i].apply(convert_str_to_list)
        else:
            df = pd.read_csv(self.config['file_path'],sep=sep)
            training_set, test_set = self.process_df(df, training_set, test_set)
        print('***********<<<<<training set********************')
        print(training_set.head(10))
        print('***********<<<<test set********************')
        print(test_set.head(10))
            
        return training_set, test_set

    def process_df(self, df, origin_training_set=None, origin_test_set=None):
        nan_value = utili.get_table_value(self.config, 'nan_value', None)
        if nan_value is None:
            df = df.dropna()
        else:
            def trans_nan(e):
                if e is None:
                    return nan_value 
                else:
                    return e
            df[self.label_key] = df[self.label_key].apply(trans_nan)

        df[self.label_key] = df[self.label_key].astype(str)
        utili.print_debug_info(df, 'after drop na', print_head = True)
        level_num = self.config['level_num']
        
        if self.config['drop_multilabel'] > 0:
            df = df[df[self.label_key].apply(lambda x: hierarchical_learning.multilabel_labels_not_greater(x, self.config['drop_multilabel']))]

        drop_insufficient_length_label = utili.get_table_value(self.config, 'drop_insufficient_length_label', None)
        if drop_insufficient_length_label:
            df[self.label_key]= df[self.label_key].apply(lambda x:hierarchical_learning.get_label_text_list(x))
        else:
            dummy_value = utili.get_table_value(self.config, 'dummy_value', None)
            df[self.label_key]= df[self.label_key].apply(lambda x:hierarchical_learning.get_label_text_list(x, level_num=level_num, dummy=dummy_value))
            df = df[df[self.label_key].apply(lambda x:len(x)>0)]
            
        if self.config['max_len'] > 0:
            df = df[df['Sequence'].apply(lambda x:len(x)<=self.config['max_len'])]
            utili.print_debug_info(df, 'after drop seq more than %d ' % self.config['max_len'], print_head = True)

        df = self._apply_threshold(df)

        utili.print_debug_info(df, 'after apply threshold', print_head = True)
        
        for i in range(level_num):
            df = self.get_level_labels(df, i, self.config['class_maps'])
            utili.print_debug_info(df, 'after select to level %d' % i, print_head = True)
        
        self.config['max_category'] = []
        for i in range(level_num):
            df, temp_max_category, temp_field_map_to_number = enzyme_data_manager.create_label_from_field(df, self.config['class_maps'],'level%d' % i, 'level%d' % i, i)
            self.config['max_category'].append(temp_max_category)
            self.config['field_map_to_number'][i] = temp_field_map_to_number
            utili.print_debug_info(df, 'after create task label to level %d' % i, print_head = True)
        print('max_category:', self.config['max_category'])
        
        if self.config['print_statistics']:
            print('following statistics information is based on data to use.')
            for index in range(level_num):
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
                        print('level %d: %d class only have %d examples' % (index, map_cnt[i], i))
                print('*level %d: %d classes less than 10, occupy %f%% of %d' % (index, less_than_10, float(less_than_10) * 100.0 / self.config['max_category'][index], self.config['max_category'][index]))
        
        
        df = df.sample(frac=self.config['fraction'])
        utili.print_debug_info(df, 'after sampling frac=%f' % self.config['fraction'])
        self.config['using_set_num'] = df.shape[0]
        #df = df.reindex(np.random.permutation(df.index))
        
        self.config['max_len'] = 0
        def set_max_len(x):
            if self.config['max_len'] <len(x):
                self.config['max_len'] = len(x)
        
        print('train_percent:%f' % self.config['train_percent'])

        target_level = self.config['target_level']
        print('target_level:', target_level)

        df['Sequence'].apply(lambda x:set_max_len(x))
        print('max_len:', self.config['max_len'])

        if origin_training_set is None or origin_test_set is None:
            print('*************normal split processs*****************')
            training_amount = int(self.config['using_set_num'] * self.config['train_percent'])
            index_name = 'level%d' % (target_level)
            training_set, test_set = data_spliter.at_least_one_label_in_test_set(df, self.config['train_percent'], index_name, self.config['max_category'][self.config['target_level']])
        else:
            print('*****************begin to merge data***************')
            print('**********origin_training_set************')
            print(origin_training_set.head(10))
            print('**********origin_test_set************')
            print(origin_test_set.head(10))
            training_set = df.merge(origin_training_set[self.id_name], how='inner', on=self.id_name)
            test_set = df.merge(origin_test_set[self.id_name], how='inner', on=self.id_name)
            print('**************df*********************')
            print(df.head(10))
            print('**********training_set************')
            print(training_set.head(10))
            print('**********test_set************')
            print(test_set.head(10))
            print('shapes..................')
            print(df.shape)
            print(origin_training_set.shape)
            print(origin_test_set.shape)
            print(training_set.shape)
            print(test_set.shape)


        if 'save_data' in self.config:
            print('save_data...')
            save_data = self.config['save_data']
            training_set.to_csv(save_data['train'], index=False, sep='\t')
            test_set.to_csv(save_data['test'], index=False, sep='\t')
            if 'meta' in save_data: 
                utili.save_obj(self.config, save_data['meta'])
        return training_set, test_set

    def normal_process(self, sep):
        print('normal_process')
        df = pd.read_csv(self.config['file_path'],sep=sep)
        utili.print_debug_info(df, info=True)
        return self.process_df(df)

    def get_data(self, sep='\t'):
        '''This function is used to get training data, validation data from a csv file
        '''
        training_set = None
        test_set = None

        if 'reuse_data' in self.config:
            print('reuse_data')
            training_set, test_set = self.reuse_data_process(sep=sep)
        else:
            print('none reuse_data')
            training_set, test_set = self.normal_process(sep=sep)


        feature_list = utili.GetNGrams(BioDefine.aaList, self.config['ngram'])
        self.config['max_features'] = len(feature_list) + 1
        training_set['Encode'] = training_set['Sequence'].apply(lambda x:utili.GetOridinalEncoding(x, feature_list, self.config['ngram']))
        test_set['Encode'] = test_set['Sequence'].apply(lambda x:utili.GetOridinalEncoding(x, feature_list, self.config['ngram']))

        utili.print_debug_info(training_set, "training set", print_head=True)
        utili.print_debug_info(test_set, "test set", print_head=True)

        self.training_set = training_set
        self.test_set = test_set
        
        x_train = self.get_x_from_df(training_set)
        y_train = self.get_y_from_df(training_set)
        x_test = self.get_x_from_df(test_set)
        y_test = self.get_y_from_df(test_set)

        task_num = self.get_task_num() 
        if task_num == 1:
            target_level = self.config['target_level']
            y_train = [y_train[target_level]]
            y_test = [y_test[target_level]]

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train, x_test, y_test

    def get_training_and_test_set(self):
        return self.training_set, self.test_set


    def get_training_data(self):
        return self.x_train, self.y_train
    
    def get_test_data(self):
        return self.x_test, self.y_test

    def get_task_num(self):
        return self.config['task_num']

    def get_max_category(self):
        ret = self.config['max_category']
        if self.get_task_num() == 1:
            ret = [self.config['max_category'][self.config['target_level']]]
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
            res.append(temp)
        return res

    def one_hot_to_labels(self, y):
        '''this function is used to transfer one-hot-encoding to label values
        '''
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
            level_num = self.config['level_num']
            i = level_num
            if not i in number_to_field:
                number_to_field[i] = utili.switch_key_value(self.config['field_map_to_number'][i])
            ret.append(self._one_hot_to_labels(y[0], i, number_to_field[i]))
        return ret

    def get_encode_info(self):
        return self.config

    def get_class_statistic(self, c):
        '''This function is used to get class statistics
        '''
        class_maps = self.config['class_maps']
        for k in class_maps:
            if c in class_maps[k]:
                cnt = class_maps[k][c]
                return cnt, k+1
                 
    def get_x(self, df):
        '''This function is used to get data used for prediction from a pandas frame
        '''
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
        '''This function is used to get data used for prediction from a file
        '''
        df = pd.read_csv(file_name, sep='\t')
        return self.get_x(df) 

def create(config):
   return enzyme_data_manager(config) 

data_manager_creator.instance.register(enzyme_data_manager.name, create) 
