import sys
import pandas as pd



def get_cluster_info(cluster_info_file, origin_file, dest_file):
    df_cluster = pd.read_csv(cluster_info_file, chunksize=300, sep='\t')
    entry_cluster_mapping = {}
    
    for records in df_cluster:
        for i in range(len(records)):
            row = records.iloc[i]
            name = row['Cluster name']
            members = row['Cluster members'].split(';')
            for member in members: 
                entry_cluster_mapping[member] = name

    df_data = pd.read_csv(origin_file, chunksize=300, sep='\t')
    
    need_head = True
    for records in df_data:
        temp_list = []
        for i in range(len(records)):
            row = records.iloc[i] 
            try:
                temp_list.append(entry_cluster_mapping[row['Entery']])
            except:
                temp_list.append('unknown')
        records['Cluster name'] = temp_list
        records.to_csv(dest_file, index=False, mode='a', sep='\t', header=need_head)
        need_head = False



if __name__=='__main__':
    get_cluster_info(sys.argv[1],sys.argv[2],sys.argv[3])

