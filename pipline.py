import enzyme_classifier
import enzyme_protein_classifier
import pandas as pd
import sys, getopt
import os

def run(input_file, output_file,is_enzyme_model, enzyme_model):
    temp_output_1 = '__temp1__.csv'
    temp_output_2 = '__temp2__.csv'
    temp_output_3 = '__temp3__.csv'
    enzyme_protein_classifier.run(input_file, temp_output_1, is_enzyme_model) 
    intermedia_cp = pd.read_csv(temp_output_1, sep='\t')
    intermedia_cc = intermedia_cp[intermedia_cp['is enzyme'].apply(lambda x:x.strip()=='Y')]
    input_df = pd.read_csv(input_file, sep='\t')
    intermedia_cc = pd.concat([intermedia_cc, input_df], axis=1, join='inner')
    intermedia_cc.to_csv(temp_output_2, sep='\t')
    enzyme_classifier.run(temp_output_2, temp_output_3, enzyme_model)
    intermedia_cc = pd.read_csv(temp_output_3, sep='\t')
    final_result = pd.merge(intermedia_cp, intermedia_cc, how='left', on='Entry name')
    final_result.to_csv(output_file, sep='\t')
    os.remove(temp_output_1)
    os.remove(temp_output_2)
    os.remove(temp_output_3)

def command_line_parser(argv):
    name = argv[0]
    argv = argv[1:]
    help_str = '%s -i <input_file> -o <output_file> -d <distinguishmodel> -e <enzymemodel>' % name
    input_file = '' 
    output_file= '' 
    is_enzyme_model = '' 
    enzyme_model = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:d:e:",["ifile", "ofile=", "distinguishmodel=", "enzymemodel="])
    except getopt.GetoptError:
        print(help_str) 
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_str)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in ('-e', '--enzymemodel'):
            enzyme_model = arg
        elif opt in ('-d', '--distinguishmodel'):
            is_enzyme_model = arg

    if not input_file or not output_file or not enzyme_model or not is_enzyme_model:
        print(help_str)
        sys.exit()
    return input_file, output_file, is_enzyme_model, enzyme_model

def main(argv):
    input_file, output_file, is_enzyme_model, enzyme_model = command_line_parser(argv)
    run(input_file, output_file, is_enzyme_model, enzyme_model)

if __name__ == "__main__":
   main(sys.argv)
