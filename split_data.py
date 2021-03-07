import pandas as pd
import sys 
def split(in_file, number, out_file, out_file2=None):
    df = pd.read_csv(in_file, sep='\t')
    df1 = df.iloc[:number+1]
    df1.to_csv(out_file, sep='\t', index=False)
    if out_file2:
        df2 = df.iloc[number + 1:]
        df2.to_csv(out_file2, sep='\t', index=False)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        split(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) == 5:
        split(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        print('in_file, number, out_file, out_file2=None')
