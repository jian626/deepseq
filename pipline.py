import enzyme_classifier
import enzyme_protein_classifier
import pandas as pd
import sys, getopt
import os

def run(input_file, output_file, is_enzyme_model, enzyme_model):
    print(input_file, output_file, is_enzyme_model, enzyme_model)
    intermedia = '__temp__.csv'
    intermedia2 = '__temp2__.csv'
    enzyme_protein_classifier.run(input_file, intermedia, is_enzyme_model) 
    intermedia_df = pd.read_csv(intermedia, sep='\t')
    intermedia_df = intermedia_df[intermedia_df['is enzyme'].apply(lambda x:x.strip()=='Y')]
    input_df = pd.read_csv(input_file, sep='\t')
    intermedia_df = pd.concat([intermedia_df, input_df], axis=1, join='inner')
    intermedia_df.to_csv(intermedia2, sep='\t')
    enzyme_classifier.run(intermedia2, output_file, enzyme_model)
    os.remove(intermedia)
    os.remove(intermedia2)

def command_line_parser(argv):
    name = argv[0]
    argv = argv[1:]
    help_str = '%s -i <input_file> -o <output_file> -d <distinguishmodel> -e <enzymemodel>' % name
    input_file = '' 
    output_file = '' 
    is_enzyme_model = '' 
    enzyme_model = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:e:d:",["ifile=","ofile=", "enzymemodel=", "distinguishmodel="])
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

    if not output_file or not input_file or not enzyme_model or not is_enzyme_model:
        print(help_str)
        sys.exit()
    return input_file, output_file, is_enzyme_model, enzyme_model

def main(argv):
    input_file = ''
    output_file = ''
    is_enzyme_model = ''
    enzyme_model = ''
    input_file, output_file, is_enzyme_model, enzyme_model = command_line_parser(argv)
    run(input_file, output_file, is_enzyme_model, enzyme_model)

if __name__ == "__main__":
   main(sys.argv)
