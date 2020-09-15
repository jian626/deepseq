import numpy as np
import pickle

n_gram_map = {}

ordinal_map = {}

debug = True 

basic_info_cnt = 0

def set_debug_flag(flag):
    debug = flag

def GetOridinalEncoding(seq, featureList, n):
    aaHash = None 
    featurelist_hash_str = '_'.join(featureList)
    if not featurelist_hash_str in ordinal_map:
        aaHash = {}
        for index, aa in enumerate(featureList):
        	aaHash[aa] = index + 1
        ordinal_map[featurelist_hash_str] = aaHash
    else:
        aaHash = ordinal_map[featurelist_hash_str]
    ret = []
    for l in range(len(seq)-n):
    	ret.append(aaHash[seq[l:l+n]])
    
    return np.asarray(ret)

def GetOneHotEncoding(seq, featureList, max_len):
    aaHash = {}
    for index, aa in enumerate(featureList):
    	aaHash[aa] = index
    temp = []
    ret = np.zeros((max_len, len(featureList)))
    for pos, l in enumerate(seq):
        try:
            index = aaHash[l]
        except:
        	print('\nseq:', seq, "\nletter:", l)
        ret[pos][index] = 1.0
    return ret


def explode(res, joined, groups):
    for i in range(len(groups) + 1):
        res.append(groups[:i] + joined + groups[i:])

def _GetInnerGrams(result, featureList, n, cur_pos=0, indicators=None):
    if indicators is None:
        indicators = [0] * n
    
    if cur_pos == n-1: 
        temp = []
        for i in range(len(indicators)-1):
            temp.append(featureList[indicators[i]])

        for j in range(len(featureList)):
            temp2 = temp.copy()
            temp2.append(featureList[j])
            result.append(''.join(temp2))
    else:
        for i in range(len(featureList)):
            indicators[cur_pos] = i
            _GetInnerGrams(result, featureList, n, cur_pos + 1, indicators)
    return result 
    

def GetNGrams(featureList, n):
    feature_hash = '%s%d' % ('_'.join(featureList), n)
    if not feature_hash in n_gram_map:
        result = []
        result = _GetInnerGrams(result, featureList, n)
        n_gram_map[feature_hash] = result
    return n_gram_map[feature_hash]

def GetNGramEncoding(seq, featureList, n, max_len):
    ngram_list = GetNGrams(featureList, n)
    return GetOridinalEncoding(seq, ngram_list, max_len)

def print_debug_info(df, title="basic info", info=False, print_head=False):
    global basic_info_cnt 
    if debug:
        print("==================%s==========info num:%d============" % (title, basic_info_cnt))
        if info:
            print('info:', df.info())
        print('shape:',df.shape)
        if print_head:
            print(df.head(10))
        print("==============================================")
        basic_info_cnt += 1

def switch_key_value(a_dict):
    return {v:k for k, v in a_dict.items()}

def map_label_to_class(map_table, label):
    ret = [] 
    for i in range(len(label)):
        if label[i]:
            ret.append(map_table[i])
    return ret
    

def save_obj(obj,name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def strict_identical_compare(L1, L2):
    v = L1 == L2
    return bool(v.all())
    
def strict_compare_report(label_set1, label_set2, length):
    res = 0
    for i in range(length):
        if strict_identical_compare(label_set1[i], label_set2[i]):
            res += 1
    return res

def get_table_value(table, key, default=None):
    ret = None
    if key in table:
        ret = table[key]
    else:
        ret = default
    return ret

if __name__ == '__main__':
    pass
		
		
		 

