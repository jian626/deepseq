import numpy as np
import pandas as pd
import yaml
import sys
from framework import init 
from framework import utili
from datetime import datetime
from sklearn.metrics import classification_report
from framework.evaluator_manager import evaluator_manager_creator 
from framework.data_manager import data_manager_creator 
from framework.model_manager import model_manager_creator
from framework.evaluator import evaluator_creator

def run(config_path):
    transfor_learning= False
    utili.set_debug_flag(False)
    config = None
    with open(config_path) as f:
        config = yaml.load(f)

    data_config = config['data_config']
    model_config = config['model_config']
    evaluator_manager_config = config['evaluator_manager_config']

    dm = data_manager_creator.instance.create(data_config)
    x_train, y_train, x_test, y_test = dm.get_data(sep='\t')
    mc = model_manager_creator.instance.create(dm, model_config)
    mc.create_model()
    ee = evaluator_creator.instance.create('enzyme_evaluator',dm)
    evaluator_list = []
    evaluator_list.append(ee)
    me = evaluator_manager_creator.instance.create(evaluator_manager_config, dm, mc, evaluator_list)
    me.evaluate()

if __name__ == '__main__':

    if len(sys.argv) < 1:
        print('please input configure file')
        return
    config_path = sys.argv[1]
    run(config_path)
    
