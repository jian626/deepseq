import numpy as np
import pandas as pd
from framework import utili
from datetime import datetime
from framework.estimator_manager import estimator_manager_creator
    
class common_estimator_manager:
    name = 'common_estimator_manager'
    def __init__(self, config, data_manager, model_manager, estimators):
        self.config = config
        self.data_manager = data_manager 
        self.model_manager = model_manager 
        self.estimators = estimators

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
            
        if not cur_round is None:
            print('***************current runing is based on %d round, this run will has %d epochs.****************' % (cur_round, epochs))

        self.model_manager.fit(x_train, y_train, epochs, batch_size)

        suffix = ''
        if not cur_round is None:
            suffix += '_round_' + str(cur_round)

        self.model_manager.save_model(suffix)
            
        y_pred = self.model_manager.predict(x_test)

        for estimator in self.estimators:
            estimator.estimate(y_pred, y_test, len(x_test), self.config['print_report'])
        return 

def create(config, data_manager, model_manager, estimator_list):
    return common_estimator_manager(config, data_manager, model_manager, estimator_list)

estimator_manager_creator.instance.register(common_estimator_manager.name, create)
