import numpy as np
import pandas as pd
from framework import utili
from datetime import datetime
from framework.evaluator_manager import evaluator_manager_creator
from framework.tools.sequence_sampling import SequenceGenerator
    
class common_evaluator_manager:
    name = 'common_evaluator_manager'
    def __init__(self, config, data_manager, model_manager, evaluators):
        self.config = config
        self.data_manager = data_manager 
        self.model_manager = model_manager 
        self.evaluators = evaluators
        self.sg = None 

    def evaluate(self):
        begin = datetime.now()
        current_time = begin.strftime("%H:%M:%S")
        print("*** estimation begin time***:", current_time)

        epochs = self.config['epochs']
        batch_round = utili.get_table_value(self.config, 'batch_round')

        x_train, y_train = self.data_manager.get_training_data()
        x_test, y_test = self.data_manager.get_test_data()

        task_num = self.data_manager.get_task_num()

        self.model_manager.compile()

        if self.config['print_summary']:
            print(self.model_manager.get_summary())

        if not self.config['train_model']:
            return

        if batch_round:
            round_size = utili.get_table_value(self.config, 'round_size', 10)
            total_size = (epochs + round_size - 1) // round_size
            for i in range(total_size):
                self._evaluate(x_train, y_train, x_test, y_test, round_size, i)
        else:
            self._evaluate(x_train, y_train, x_test, y_test, epochs)

        end = datetime.now()
        current_time = end.strftime("%H:%M:%S")
        print("***end time***:", current_time)
        print("total estimation time cost:", end - begin)
        print("========================done==========================")

    def _evaluate(self, x_train, y_train, x_test, y_test, epochs, cur_round = None):

        task_num = self.data_manager.get_task_num()
        batch_size = self.config['batch_size']

        if not self.sg:
            self.sg = SequenceGenerator(self.data_manager, batch_size)
            
        if not cur_round is None:
            print('***************current runing is based on %d round, this run will has %d epochs.****************' % (cur_round, epochs))

        self.model_manager.fit_generator(self.sg)
        #self.model_manager.fit(x_train, y_train, epochs, batch_size)

        suffix = ''
        if not cur_round is None:
            suffix += '_round_' + str(cur_round)

        self.model_manager.save_model(suffix)
            
        y_pred = self.model_manager.predict(x_test)

        for evaluator in self.evaluators:
            evaluator.evaluate(y_pred, y_test, len(x_test), self.config['print_report'])
        return 

def create(config, data_manager, model_manager, evaluator_list):
    return common_evaluator_manager(config, data_manager, model_manager, evaluator_list)

evaluator_manager_creator.instance.register(common_evaluator_manager.name, create)
