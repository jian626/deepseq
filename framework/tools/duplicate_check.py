import pandas as pd
import sys


def check_df(df1, df2, key_name):
    pass


def check_file(file1, file2, key_name, sep='\t'):
    df1 = pd.read_csv(file1, sep='\t') 
    df2 = pd.read_csv(file2, sep='\t')

    dict_1 = {}
    result = 0

    for key in df1[key_name]:
        if key in dict_1:
            dict_1[key] += 1
        else:
            dict_1[key] = 1

    for key in df2[key_name]:
        if key in dict_1:
            result += 1
    return len(df1), len(df2), result, result / len(df1), result / len(df2) 


if __name__ == '__main__':
    print(check_file(sys.argv[1], sys.argv[2], sys.argv[3]))
