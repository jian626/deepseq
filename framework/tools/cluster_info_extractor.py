import pandas as pd
import random
import numpy as np
import sys

def get_key_cluster_name(name):
    _, name = name.split('_')
    name = name.strip()
    while name[-1].isnumeric():
        name = name[:-1]
    return name


def cluster(file_name, out_file, key_cluster_name, train_percentage):
    clusters = {}
    sep = '\t'
    df = pd.read_csv(file_name, sep=sep)
    for index in range(len(df)):
        r = df.iloc[index]
        name = get_key_cluster_name(r['Entry name'])
        if not name in clusters:
            clusters[name] = [] 
        same_cluster = clusters[name]
        same_cluster.append(index)

    cluster_info = np.array([None] * len(df))
    for (k, v) in clusters.items():
        cluster_info[v] = [k] * len(v)
    df[key_cluster_name] = cluster_info
        
    print('len of clusters', len(clusters))
    print(df.head(20))
    if train_percentage:
        train_len = int(len(clusters) * float(train_percentage))
        cluster_names = list(clusters.keys())
        random.shuffle(cluster_names)
        training_clusters = cluster_names[:train_len]
        test_clusters = cluster_names[train_len:]
        dest = []
        for name in training_clusters:
            dest += clusters[name]
        df.iloc[dest].to_csv('train_' + out_file, sep=sep, index = False)
        dest = []
        for name in test_clusters:
            dest += clusters[name]
        df.iloc[dest].to_csv('test_' + out_file, sep=sep, index = False)
    else:

        df.to_csv(out_file, sep=sep, index=False)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        cluster(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 5:
        cluster(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

            

    
