import numpy as np

n_gram_map = {}

ordinal_map = {}

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

if __name__ == '__main__':
    #r = GetOneHotEncoding('ACE')
    in_str = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    res = []
    r = _GetInnerGrams(res, in_str, 4)
    print(r, len(r), len(in_str))
		
		
		
		 
