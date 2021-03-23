import pandas as pd
import math
import sys



def get_function(raw_function_str):
    #raw_function_str = raw_function_str.strip().split(';')
    #raw_function_str.sort()
    #raw_function_str = raw_function_str.strip().split(';')
    #raw_function_str.sort()
    return raw_function_str

def get_entropy(cluster):
    n = len(cluster[0])
    res = 0
    for k, v in cluster[1].items():
        res += -1 * v/n * math.log(v/n)
    if len(cluster[1])>1:
        print('cluster:', cluster[1])
    return res

def get_ex_entropy(cluster, total):
    n = len(cluster[0])
    return n / total * get_entropy(cluster)

def get_entropy_from_df(df, id_entry, cluster_entry, function_entry):
    df =df.dropna()
    total = df.shape[0]
    clusters = {}
    for index in range(total):
        row = df.iloc[index]
        cluster_name = row[cluster_entry]
        id_name = row[id_entry]
        raw_function_str = str(row[function_entry])
        if raw_function_str == 'nan':
            raw_function_str = ''
        function = get_function(raw_function_str)
        cluster = None 
        if not cluster_name in clusters:
            clusters[cluster_name] = ([], {})
        cluster = clusters[cluster_name]
        cluster[0].append(id_name)
        function_str = str(function)
        if function_str in cluster[1]:
            cluster[1][function_str] += 1
        else:
            cluster[1][function_str] = 1

    entropy_res = 0
    diff_num = 0
    for name, cluster in clusters.items():  
        if len(cluster[0]) > 1:
            diff_num += 1
        ex_entropy = get_ex_entropy(cluster, total)
        entropy_res += ex_entropy
    print('diff_num:', diff_num)
    print('len clusters:', len(clusters))
    print('proportion:', diff_num / len(clusters))
    return entropy_res
        
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('file_name, id_entry, cluster_entry, function_entry')
        sys.exit(0)
    file_name = sys.argv[1]
    id_entry = sys.argv[2]
    cluster_entry = sys.argv[3]
    function_entry = sys.argv[4]
    df = pd.read_csv(file_name, sep='\t')
    entropy = get_entropy_from_df(df, id_entry, cluster_entry, function_entry)
    print('the entrop is:', entropy)

    


