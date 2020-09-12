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

begin = datetime.now()
current_time = begin.strftime("%H:%M:%S")
print("begin time:", current_time)

fraction = 1 
number = 40000
max_features = len(BioDefine.aaList) + 1
embedding_dims = 16 
filters = 64 
delta = 24 
kernelSize = 3 
hidden1Dim = 256 
hidden2Dim = 256
strides = 1 
pool_size = 2 
layerLen = 5 
convLen = 5 
poolLen = 1 
epochs = 1000 
max_len = 1000
train_percent = 0.7
batch_size = 400
basic_info_cnt = 0 
ec_level = 4 
compare_level = 2 
main_task = 3 
train_model = True 
drop_na = True
drop_multilabel = False 
multi_task = False 
apply_dummy_label = False 
minority_threshold = 5
ngram = 3 


class_maps = { 
        0:{},
        1:{},
        2:{},
        3:{}
   } 


def print_basic_info(df, title="basic info", info=False, print_head=False):
    global basic_info_cnt
    print("==================%s==========cnt:%d============" % (title, basic_info_cnt))
    if info:
        print('info:', df.info())
    print('shape:',df.shape)
    if print_head:
        print(df.head(10))
    print("==============================================")
    basic_info_cnt += 1

def get_ec_list(ec):
    if ec:
        return ec.split(';')
    else:
        return None 

def get_ec_level_list(ec, level):
    ret = [] 
    ec_list = get_ec_list(ec)
    for e in ec_list:
        if has_level(level, e): 
            ret.append(e)
    return ret
    

def get_ec(ec, level):
    ret = None
    if type(ec) == list:
        ret = []
        for e in ec:
            res = get_ec(e, level)
            ret.append(res)
        ret = list(set(ret))
    else:
        ret = '' 
        if ec:
            l = ec.split('.')
            ret = str(int(l[0]))
            for i in range(1, level+1):
                try:
                    ret = ret + '.' + str(int(l[i]))
                except:
                    ret = ret + '.unknown'
            if ret in class_maps[level]:
                class_maps[level][ret] += 1
            else:
                class_maps[level][ret] = 1
                
    return ret

def not_multilabel_enzyme(ec):
    if ec:
        l = ec.split(';')
        if len(l)>1:
            return False 
    return True

def get_example_label_coverage(df):
    ret = {}
    for _, row in df.iterrows():
        if row['Lables'] in ret:
            ret[row['Lables']] += 1
        else:
            ret[row['Lables']] = 1
    ret = {k: v for k, v in sorted(ret.items(), key=lambda item: item[1])}
    return ret

def print_coverage_table(coverage, num=10):
    print('-------print coverage table---------cnt:%d---------' % num)
    for i, k in enumerate(coverage):
        if i >= num:
            break
        print(k, coverage[k])
        
def create_map(m):
    ret = {}
    for v, k in enumerate(m):
        ret[k] = v
    return ret

def has_level(level, ec):
    l = ec.split('.')
    if len(l) < level:
        return False 
    try:
        for e in l:
            int(e)
    except:
        return False
    return True

def get_level_labels(df, level):
    df["level%d" % level] = df['EC number'].apply(lambda x:get_ec(x, level))
    return df

def map_ec_to_value(ec, map_table):
    ret = None
    if type(ec) == list:
        ret = []
        for e in ec:
            ret.append(map_table[e])
    else:
        ret = map_table[ec]
    return ret

def create_label_from_field(df, field_name, label_name, i):
    values = list(class_maps[i].keys())
    field_map_to_number = create_map(values)
    df[label_name] = df[field_name].apply(lambda x:map_ec_to_value(x, field_map_to_number))
    return df,len(field_map_to_number) 

def create_input_embedding(max_len, max_features, embedding_dims):
    inputLayer = Input(shape=(max_len,))
    return inputLayer, Embedding(max_features,
                        embedding_dims,
                        input_length=max_len)(inputLayer)

def map_number_to_label(ec_set, num_classes, drop_multilabel):
    ret = None
    if not drop_multilabel:
        ret = np.zeros((len(ec_set), num_classes)) 
        for index, ec in enumerate(ec_set):
            for e in ec:
                ret[index][e] = 1
    else:
        ret = to_categorical(ec_set, num_classes=num_classes)
    return ret

def get_data_and_label(data_set):
    x = data_set['Encode']
    x = sequence.pad_sequences(x, maxlen=max_len, padding='post')
    y = []
    for i in range(ec_level):
        temp = map_number_to_label(data_set['task%d' % i], max_category[i], drop_multilabel)
        y.append(temp)
    return x, y

def create_public_net(input_layer, hidden_dense_len):
    lastLayer = input_layer
    
    lastLayer_1 = lastLayer
    lastLayer_2 = lastLayer
    lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Dropout(0.2)(lastLayer_1)
    lastLayer_1 = AveragePooling1D(pool_size=pool_size, strides=strides, padding='same')(lastLayer_1)
    lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Dropout(0.2)(lastLayer_1)
    lastLayer_1 = AveragePooling1D(pool_size=pool_size, strides=strides, padding='same')(lastLayer_1)
    lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Dropout(0.2)(lastLayer_1)
    lastLayer_1 = AveragePooling1D(pool_size=pool_size, strides=strides, padding='same')(lastLayer_1)
    lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(48, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(64, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Conv1D(16, kernelSize, padding='same', activation='relu')(lastLayer_1)
    lastLayer_1 = Dropout(0.2)(lastLayer_1)
    lastLayer_1 = MaxPooling1D(pool_size=pool_size, strides=strides, padding='same')(lastLayer_1)
    mainLayer = tf.keras.layers.Add()([lastLayer_1, lastLayer_2])
    lastLayer_3 = mainLayer
    mainLayer = Conv1D(48, kernelSize, padding='same', activation='relu')(mainLayer)
    mainLayer = Conv1D(48, kernelSize, padding='same', activation='relu')(mainLayer)
    mainLayer = Conv1D(64, kernelSize, padding='same', activation='relu')(mainLayer)
    mainLayer = Conv1D(64, kernelSize, padding='same', activation='relu')(mainLayer)
    mainLayer = MaxPooling1D(pool_size=pool_size, strides=strides, padding='same')(mainLayer)
    mainLayer = Dropout(0.2)(mainLayer)
    mainLayer = Conv1D(48, kernelSize, padding='same', activation='relu')(mainLayer)
    mainLayer = Conv1D(48, kernelSize, padding='same', activation='relu')(mainLayer)
    mainLayer = Conv1D(64, kernelSize, padding='same', activation='relu')(mainLayer)
    mainLayer = Conv1D(64, kernelSize, padding='same', activation='relu')(mainLayer)
    mainLayer = MaxPooling1D(pool_size=pool_size, strides=strides, padding='same')(mainLayer)
    mainLayer = Dropout(0.2)(mainLayer)
    lastLayer = tf.keras.layers.Concatenate()([mainLayer, lastLayer_3])
    
    lastLayer = Dropout(0.2)(lastLayer)
    lastLayer = Flatten()(lastLayer)
    lastLayer = Dense(256)(lastLayer)
    lastLayer = Dropout(0.2)(lastLayer)
    return lastLayer

level_map_to_number = {}            
    
myfile = 'uniprot-reviewed_yes.tab'
df = pd.read_csv(myfile,sep='\t')
print_basic_info(df, info=True)

if drop_na:
    df = df.dropna()

df.drop(labels=['Entry', 'Entry name'], axis=1)

df['EC number'] = df['EC number'].astype(str)
print_basic_info(df, 'after drop na', print_head = True)

if drop_multilabel:
    df = df[df['EC number'].apply(lambda x:not_multilabel_enzyme(x))]
    print_basic_info(df, 'after drop multilabel')
    if not apply_dummy_label:
        print_basic_info(df, 'before drop dummy')
        df = df[df['EC number'].apply(lambda x:has_level(ec_level, x))]
        print_basic_info(df, 'after drop dummy')
else:
    if not apply_dummy_label:
        df['EC number']= df['EC number'].apply(lambda x:get_ec_level_list(x, ec_level))
        df = df[df['EC number'].apply(lambda x:len(x)>0)]
    else:
        df['EC number']= df['EC number'].apply(lambda x:get_ec_list(x))
        
    df['EC count'] = df['EC number'].apply(lambda x:len(x))

#if minority_threshold > 0:
    

if max_len > 0:
    df = df.drop(df[df['Length']>max_len].index)
    print_basic_info(df, 'after drop seq more than %d ' % max_len, print_head = True)

for i in range(ec_level):
    df = get_level_labels(df, i)
    print_basic_info(df, 'after select to level %d' % i, print_head = True)

sorted_k = {k: v for k, v in sorted(class_maps[ec_level-1].items(), key=lambda item: item[1])}
print(sorted_k)


max_category = []
for i in range(ec_level):
    df, temp_max_category = create_label_from_field(df, 'level%d' % i, 'task%d' % i, i)
    max_category.append(temp_max_category)
    print_basic_info(df, 'after create task label to level %d' % i, print_head = True)
print('max_category:', max_category)

df = df.sample(frac=fraction)
print_basic_info(df, 'after sampling frac=%f' % fraction)
using_set_num = df.shape[0]
df = df.reindex(np.random.permutation(df.index))

max_len= max(df['Length'])
print('max_len:', max_len)
#df['Encode'] = df['Sequence'].apply(lambda x:utili.GetOridinalEncoding(x, BioDefine.aaList, max_len))
feature_list = utili.GetNGrams(BioDefine.aaList, ngram)
max_features = len(feature_list) + 1
df['Encode'] = df['Sequence'].apply(lambda x:utili.GetOridinalEncoding(x, feature_list, ngram))


print('train_percent:%f' % train_percent)
training_set = df.iloc[:int(using_set_num * train_percent)]
test_set = df.iloc[training_set.shape[0]:]
print_basic_info(training_set, "training set", print_head=True)
print_basic_info(test_set, "test set", print_head=True)

x_train, y_train = get_data_and_label(training_set)
x_test, y_test = get_data_and_label(test_set)

if train_model:
    output = []
    input_embedding_layer, lastLayer = create_input_embedding(max_len, max_features, embedding_dims)
    lastLayer = create_public_net(lastLayer, hidden1Dim)

    task_loss_num = 1
    train_target = None 
    test_target = None 
    if multi_task:
        for i in range(ec_level):
            task_lastLayer = Dense(hidden2Dim)(lastLayer)
            task_lastLayer = Dense(max_category[i], activation='sigmoid', name="task_%d_1" % i)(task_lastLayer)
            output.append(task_lastLayer)
        task_loss_num = ec_level
        train_target = y_train
        test_target = y_test
    else:
        lastLayer = Dense(max_category[ec_level-1], activation='softmax')(lastLayer)
        output.append(lastLayer)
        train_target = y_train[ec_level-1]
        test_target = y_test[ec_level-1]
        task_loss_num = 1
    
    optimizer = Adam()
    model = Model(inputs=input_embedding_layer, outputs=output)
    #model = Model(inputs=input_embedding_layer, outputs=lastLayer)
    #model.compile(optimizer=optimizer, loss=['categorical_crossentropy'] * task_loss_num , metrics=['categorical_accuracy'])
    model.compile(optimizer=optimizer, loss=['binary_crossentropy'] * task_loss_num, metrics=['categorical_accuracy'])
    #model.compile(optimizer=optimizer, loss=['categorical_crossentropy'] * ec_level , metrics=['categorical_accuracy'])
    print(model.summary())
    
    callback = tf.keras.callbacks.EarlyStopping(monitor=['val_categorical_accuracy'], restore_best_weights=True, patience=5, verbose=1)
    
    history = model.fit(x_train, train_target, epochs=20,  batch_size=batch_size, validation_split=1/6)
    #history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=1/6, callbacks=[callback])
    model.save_weights('model.h5')
    
    print('==============history=============')
    loss = model.evaluate(x_test, test_target)
    print('loss:',loss)
    y_pred = model.predict(x_test)
    print(y_pred)
    if multi_task:
        y_pred = (y_pred[compare_level - 1] > 0.5)
        test_target = test_target[compare_level-1]
    else:
        y_pred = (y_pred > 0.5)
    print('y_pred:')
    print(y_pred)
    print('y_pred shape:', y_pred.shape)
    report = classification_report(test_target, y_pred)
    print('report:')
    print(report)
    
    end = datetime.now()
    current_time = end.strftime("%H:%M:%S")
    print("end time:", current_time)
    print("end - begin:", end - begin)
    print("========================done==========================")
