import enzyme_classifier
import enzyme_protein_classifier
import pandas as pd
import sys, getopt
import os

def run(input_file, first_output, second_output, is_enzyme_model, enzyme_model):
    temp_output = '__temp2__.csv'
    enzyme_protein_classifier.run(input_file, first_output, is_enzyme_model) 
    intermedia_df = pd.read_csv(first_output, sep='\t')
    intermedia_df = intermedia_df[intermedia_df['is enzyme'].apply(lambda x:x.strip()=='Y')]
    input_df = pd.read_csv(input_file, sep='\t')
    intermedia_df = pd.concat([intermedia_df, input_df], axis=1, join='inner')
    intermedia_df.to_csv(temp_output, sep='\t')
    enzyme_classifier.run(temp_output, second_output, enzyme_model)
    os.remove(temp_output)

def command_line_parser(argv):
    name = argv[0]
    argv = argv[1:]
    help_str = '%s -i <input_file> -f <first_output> -s <second_output> -d <distinguishmodel> -e <enzymemodel>' % name
    input_file = '' 
    first_output= '' 
    second_output= ''
    is_enzyme_model = '' 
    enzyme_model = ''
    try:
        opts, args = getopt.getopt(argv,"hi:f:s:d:e:",["ifile", "firstoutput=", "secondoutput=", "distinguishmodel=", "enzymemodel="])
    except getopt.GetoptError:
        print(help_str) 
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_str)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-f", "--firstoutput"):
            first_output= arg
        elif opt in ("-s", "--secondoutput"):
            second_output = arg
        elif opt in ('-e', '--enzymemodel'):
            enzyme_model = arg
        elif opt in ('-d', '--distinguishmodel'):
            is_enzyme_model = arg

    if not input_file or not first_output or not second_output or not enzyme_model or not is_enzyme_model:
        print(help_str)
        sys.exit()
    return input_file, first_output, second_output, is_enzyme_model, enzyme_model

def main(argv):
    input_file, first_output, second_output, is_enzyme_model, enzyme_model = command_line_parser(argv)
    run(input_file, first_output, second_output, is_enzyme_model, enzyme_model)

if __name__ == "__main__":
   main(sys.argv)
