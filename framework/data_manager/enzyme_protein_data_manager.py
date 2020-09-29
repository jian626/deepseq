import pandas as pd
import numpy as np
from framework import utili 
from framework.bio import BioDefine
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from framework.data_manager import data_manager_creator

class enzyme_protein_data_manager:
    name = 'enzyme_protein_data_manager'
    def __init__(self, config):
        self.config = config
        self.config['max_category'] = 2

    def get_data(self, sep='\t'):
        df = pd.read_csv(self.config['file_path'],sep=sep)
        print("total set:", df.shape[0])
        df = df[df['Sequence'].apply(lambda seq:len(seq)<=self.config['max_len'])]
        print("after drop:", df.shape[0])
        df = df.sample(frac=self.config['fraction'])
        using_set_num = df.shape[0]
        print("using set num:", using_set_num)
        df = df.reindex(np.random.permutation(df.index))
        
        df['Lables'] = 1 
        df.loc[pd.isna(df['EC number']), 'Lables'] = 0 
        df.reset_index(inplace=True, drop=True)
        print('enzyme cnt:',df[df.Lables>0].shape[0])
        print('non-enzyme cnt:', df[df.Lables==0].shape[0])

        self.config['max_len'] = 0
        def get_len(seq):
            if self.config['max_len']< len(seq):
                self.config['max_len'] = len(seq)
            
        df['Sequence'].apply(get_len)
        
        max_len = self.config['max_len']
        print('max_len:', max_len)
        feature_list = utili.GetNGrams(BioDefine.aaList, self.config['ngram'])
        self.config['max_features'] = len(feature_list) + 1
        df['Encode'] = df['Sequence'].apply(lambda x:utili.GetOridinalEncoding(x, feature_list, self.config['ngram']))

        training_set = df.iloc[:int(using_set_num * self.config['train_percent'])]
        print('training set enzyme cnt:',training_set[training_set.Lables>0].shape[0])
        print('training non-enzyme cnt:', training_set[training_set.Lables==0].shape[0])
        test_set = df.iloc[training_set.shape[0]:]
        print("training len:", training_set.shape[0])
        print("test len:", test_set.shape[0])

        
        x_train = training_set['Encode']
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        
        y_train = training_set['Lables']
        y_train = to_categorical(y_train)
        
        x_test = test_set['Encode']
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)
        
        y_test = test_set['Lables']
        y_test = to_categorical(y_test)

        y_train = [y_train]
        y_test = [y_test]

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
        return 1 

    def get_max_category(self):
        ret = self.config['max_category']
        return [ret]

    def get_max_feature(self):
        return self.config['max_features']

    def get_max_len(self):
        return self.config['max_len']

    def one_hot_to_labels(self, y):
        ret = []
        y = y[0]
        for e in y:
            temp = 'unexpected'
            for i, c in enumerate(e):
                if c:
                    if i == 0:
                        temp = 'N'
                    else:
                        temp = 'Y'
                    break
            ret.append(temp)
        ret = [ret]
        return ret

    def get_encode_info(self):
        return self.config

    def get_class_statistic(self, c):
        pass
                 
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

def create(config):
    return enzyme_protein_data_manager(config)

data_manager_creator.instance.register(enzyme_protein_data_manager.name, create)
