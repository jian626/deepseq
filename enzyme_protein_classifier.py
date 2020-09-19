from framework.model_manager import model_manager_creator 
import numpy as np
import sys, getopt
import pandas as pd

def run(input_file, output_file, model_name):
    mc = model_manager_creator.create_from_file(model_name)
    y_pred, entry_name = mc.predict_on_file(input_file)
    data_manager = mc.get_data_manager()
    bool_labels = []
    
    task_num = data_manager.get_task_num()
    if task_num == 1:
        y_pred = [y_pred]
    
    for i in range(data_manager.get_task_num()):
        bool_labels.append(y_pred[i] > 0.5)
    
    labels = data_manager.one_hot_to_labels(bool_labels)
    labels = labels[0]
    labels = np.array(labels)
    df = pd.DataFrame(labels, columns=['is enzyme'])
    entry_name = entry_name.to_frame()
    entry_name.reset_index(inplace=True, drop=True)
    df = pd.concat([entry_name, df], axis=1)
    df.to_csv(output_file, sep='\t', index=False)

def command_line_parser(argv):
    help_str = 'enzyme_protein_classifier.py -i <input_file> -o <output_file> -m <model_name>'
    input_file = ''
    output_file = ''
    model_name = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","ofile=", "model="])
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
        elif opt in ('-m', '--model'):
            model_name = arg
    if not output_file or not input_file or not model_name:
        print(help_str)
        sys.exit()
    return input_file, output_file, model_name

def main(argv):
    input_file = ''
    output_file = ''
    model_name = ''
    input_file, output_file, model_name = command_line_parser(argv)
    run(input_file, output_file, model_name)

if __name__ == "__main__":
   main(sys.argv[1:])
