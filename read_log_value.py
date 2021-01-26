import os
import sys
import re
import pandas as pd

def get_value_in_file(file_name, field_name, field_number):
    for line in reversed(list(open(file_name))):
        if field_name in line:
            value = line.split(field_name)[1]
            value = value.strip()
            value = re.split(' +',value)
            return value[field_number]



def get_data(dir_name, field_name, field_number, sorted_file=None):
    name_list = os.listdir(dir_name)
    file_dict = {} 
    for name in name_list:
        if name.endswith('log'):
            key = re.split('_[-0-9]*_',name)[0]
            file_list = None 
            if not key in file_dict:
                file_dict[key] = [] 
            file_list = file_dict[key]
            file_list.append(name)
    dataframe = {}
    for k, v in file_dict.items():
        datas = []
        dataframe[k] = datas 
        for file_name in v:
            value = get_value_in_file(dir_name + '/'+ file_name, field_name, field_number)
            datas.append(value)
    df = pd.DataFrame(dataframe)
    print(df.head(10))
    if sorted_file:
        df.to_csv(sorted_file, index=False, sep='\t')




if __name__ == '__main__':
    if len(sys.argv) == 4:
        get_data(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) == 5:
        get_data(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    


