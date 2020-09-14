from tensorflow.keras.preprocessing import sequence
import numpy as np

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
    

def get_ec(ec, level, class_maps):
    ret = None
    if type(ec) == list:
        ret = []
        for e in ec:
            res = get_ec(e, level, class_maps)
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

def get_level_labels(df, level, class_maps):
    df["level%d" % level] = df['EC number'].apply(lambda x:get_ec(x, level, class_maps))
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

def create_label_from_field(df, class_maps, field_name, label_name, i):
    values = list(class_maps[i].keys())
    field_map_to_number = create_map(values)
    df[label_name] = df[field_name].apply(lambda x:map_ec_to_value(x, field_map_to_number))
    return df,len(field_map_to_number), field_map_to_number

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

def get_data_and_label(data_set, config):
    x = data_set['Encode']
    x = sequence.pad_sequences(x, maxlen=config['max_len'], padding='post')
    y = []
    for i in range(config['ec_level']):
        temp = map_number_to_label(data_set['task%d' % i], config['max_category'][i], config['drop_multilabel'])
        y.append(temp)
    return x, y

def get_part_level(ec, level):
    l = ec.split('.')
    if level < len(l):
        r = '.'.join(l[:level])
        if not 'unknown' in r:
            return r

def is_conflict(long_level, short_level, compare_level):
    test_map_l = [] 
    test_map_s = [] 
    for e in long_level:
        part_level = get_part_level(e, compare_level)
        if part_level:
            test_map_l.append(part_level) 

    for e in short_level:
        part_level = get_part_level(e, compare_level)
        if part_level:
            test_map_s.append(part_level) 
    return (set(test_map_l).difference(set(test_map_s)))
            
            
            
    
        

        
