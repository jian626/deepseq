import numpy as np
import sys
import re
import random
import pandas as pd
from os.path import isfile, join
from os import listdir

def get_data_from_file(file_name, field_name, level, field_num=3):
    with open(file_name) as f:
        line = f.readline()
        while line:
            if 'report level ' +  str(level) in line:
                break
            line = f.readline()

        while line:
            if field_name in line:
                line = line.strip()
                print(line)
                line = line.split(field_name)[1]
                print('res:', re.split('\s+', line))
                return float(re.findall("\d+\.*\d*", re.split('\s+', line)[field_num])[0])
            line = f.readline()

def get_log_date(dir_name, field_name, level, field_num, sep=',', ext_name='.log'):
    name_hash = {}
    for f in listdir(dir_name):
        if isfile(join(dir_name, f)) and f.endswith(ext_name):
            rf = f[::-1]
            matched_list = re.findall('(_\d+_){1}', rf)
            key = f[0: f.rfind(matched_list[0])]
            #key = re.split('_\d+_', f)
            print("key:", key)
            print("matched_list:", matched_list)
            if not key in name_hash:
                name_hash[key] = []
            name_hash[key].append(f)
    ind_name = 'index'
    data_frame = {ind_name:[]}
    coln = None
    for key, files in name_hash.items():
        files.sort()
        values = []
        for f in files:
            values.append(get_data_from_file(join(dir_name,f), field_name, level, field_num))
        mean = np.mean(values)
        var = np.var(values)
        values.append(mean)
        values.append(var)
        if coln is None:
            coln = len(files)
        data_frame[key] = values
    data_frame[ind_name] = [x for x in range(coln)] 
    data_frame[ind_name] = data_frame[ind_name] + ['mean', 'var']
    df = pd.DataFrame(data=data_frame)
    df.to_csv(field_name.replace(' ', '_') + '.csv', index=False, sep=sep)

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('dir_name, field_name, level, field_num, sep')
    else:
        get_log_date(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])

