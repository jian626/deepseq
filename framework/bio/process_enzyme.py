from tensorflow.keras.preprocessing import sequence
import numpy as np

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

def get_label_at_least_level(label, level, dummy=None):
    l = label.split('.')
    if len(l) < level:
        l +=[dummy] * (level - len(l))
    return '.'.join(l)
    

def _get_label_to_level(label, level, dummy=None):
    '''get EC numbers to the input level. when there is not enough, '-' is used. 
    '''
    l = label.split('.')
    res = []
    for index, e in enumerate(l):
        try:
            if index < level:
                e = str(int(e))
                res.append(e)
            else:
                break
        except:
            break
        
    if len(res) < level:
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
            ret.append(res)
        ret = list(set(ret))
    else:
        ret = '' 
        if label:
            ret = _get_label_to_level(label, level, dummy)
            if class_maps:
                if ret in class_maps[level-1]:
                    class_maps[level-1][ret] += 1
                else:
                    class_maps[level-1][ret] = 1
    return ret

def test_str_not_multilabel_labels(values):
    if values:
        l = values.split(';')
        if len(l)>1:
            return False 
    return True

def create_map(m):
    ret = {}
    for v, k in enumerate(m):
        ret[k] = v
    return ret

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
            
            
            
    
        

        
