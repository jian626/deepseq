from tensorflow.keras.preprocessing import sequence
import numpy as np

label_spliter = ';'
level_spliter = '.'
level_complement = '.-'

def get_label_list(value):
    if value:
        return value.split(';')
    else:
        return None 

def get_label_list_according_to_level(data, level):
    '''Get levels  which has the input level from a string
    '''
    ret = [] 
    ec_list = get_label_list(data)
    for e in ec_list:
        if has_level(level, e): 
            ret.append(e)
    return ret

def get_label_text_list(value, level_num =None, dummy=None):
    ret = []
    label_list = get_label_list(value)
    if label_list:
        if level_num:
            for label_text in label_list:
                label_text = label_text.split(level_complement)[0].strip()
                label = get_label_at_least_level(label_text, level_num, dummy)
                if label:
                    ret.append(label)
                else:
                    print('omited:', label_text)
        else:
            for label_text in label_list:
                label_text = label_text.split(level_complement)[0].strip()
                ret.append(label_text)
    return ret

def get_label_at_least_level(label, level_num, dummy=None):
    l = label.split(level_spliter)
    if len(l) < level_num:
        if dummy:
            l +=[dummy] * (level_num - len(l))
        else:
            l = [] 
    return level_spliter.join(l)
    

def _get_label_to_level(label, level, dummy=None):
    '''get EC numbers to the input level. when there is not enough, dummy is used.
        in this case, if dummy is not provide, the '' is returen
    '''
    l = label.split('.')
    res = []
    for index, e in enumerate(l):
        try:
            if index <= level:
                e = str(int(e))
                res.append(e)
            else:
                break
        except:
            break
        
    if len(res) < level + 1:
        if dummy:
            res += [dummy] * (level - len(res))
        else:
            res = [] 

    return '.'.join(res)


def get_label_to_level(label, level, dummy=None, class_maps=None):
    '''get label according to level, when there is no that level, unknow is used. and collect class information
    '''
    ret = None
    if type(label) == list:
        ret = []
        for e in label:
            res = get_label_to_level(e, level, dummy, class_maps)
            if res:
                ret.append(res)
        ret = list(set(ret))
    else:
        ret = '' 
        if label:
            ret = _get_label_to_level(label, level, dummy)
            if class_maps:
                if ret:
                    if ret in class_maps[level]:
                        class_maps[level][ret] += 1
                    else:
                        class_maps[level][ret] = 1
    return ret


def multilabel_labels_not_greater(values, threshold):
    if values:
        l = values.split(';')
        if len(l)>threshold:
            return False 
    return True

def has_level(level, label):
    l = label.split('.')
    if len(l) < level:
        return False 
    try:
        for e in l:
            int(e)
    except:
        return False
    return True

def get_part_level(label, level):
    l = label.split('.')
    if level <= len(l):
        r = '.'.join(l[:level])
        if not 'unknown' in r:
            return r

def get_conflict(long_level, short_level, compare_level):
    test_map_l = [] 
    test_map_s = [] 
    for e in long_level:
        part_level = _get_label_to_level(e, compare_level)
        if part_level:
            test_map_l.append(part_level) 

    for e in short_level:
        part_level = _get_label_to_level(e, compare_level)
        if part_level:
            test_map_s.append(part_level) 
    return set(test_map_l).difference(set(test_map_s))

if __name__ == '__main__':
    long_level = ['1.2.2.unknown', '2.2.3.4', '2.2.3.1']
    short_level = ['1.2.2', '2.2.3', ]
    s = get_conflict(long_level, short_level, 3)
    print(s)
    print(bool(s))
            
            
            
    
        

        
