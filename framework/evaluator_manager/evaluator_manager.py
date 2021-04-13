import numpy as np
import pandas as pd
from framework import utili
from datetime import datetime
from framework.evaluator_manager import evaluator_manager_creator
from framework.training.batch_generator import batch_generator_creator 
from framework.training.method import training_method_creator
#from framework.training.batch_generator.cluster_random_sampling import SequenceGenerator
#from framework.training.batch_generator.inhomogenous_cluster_training import inhomogenous_cluster_training as SequenceGenerator
#from framework.training.batch_generator.homogenous_cluster_training import homogenous_cluster_training as SequenceGenerator
    
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

        x_train, y_train, train_loss_weight = self.data_manager.get_training_data()
        x_test, y_test, test_loss_weight = self.data_manager.get_test_data()
        x_validation, y_validation, validation_loss_weight = self.data_manager.get_validation_data()

        task_num = self.data_manager.get_task_num()


        if self.config['print_summary']:
            print(self.model_manager.get_summary())

        if not self.config['train_model']:
            return

        if train_loss_weight:
            for i in range(task_num): 
                y_train[i] = np.concatenate((y_train[i], train_loss_weight[i]), axis=1)

        if validation_loss_weight:
            for i in range(task_num): 
                y_validation[i] = np.concatenate((y_validation[i], validation_loss_weight[i]), axis=1)

        if batch_round:
            round_size = utili.get_table_value(self.config, 'round_size', 10)
            total_size = (epochs + round_size - 1) // round_size
            for i in range(total_size):
                self._evaluate(x_train, y_train, x_test, y_test, epochs=round_size, cur_round=i)
        else:
            validation_data = None
            print('************************validation_data*******************')
            print(validation_data)
            if (not x_validation is None) and len(x_validation) > 0: 
                print('==============================validation data used======================')
                validation_data = (x_validation, y_validation)
            self._evaluate(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, validation_data=validation_data,  epochs=epochs, train_loss_weight=train_loss_weight, test_loss_weight=test_loss_weight)

        end = datetime.now()
        current_time = end.strftime("%H:%M:%S")
        print("***end time***:", current_time)
        print("total estimation time cost:", end - begin)
        print("========================done==========================")

    def _evaluate(self, x_train, y_train, x_test, y_test, epochs, validation_data=None, train_loss_weight=None, test_loss_weight=None, cur_round = None):

        print('len(y_train)', len(y_train))
        task_num = self.data_manager.get_task_num()
        batch_size = self.config['batch_size']

        self.model_manager.compile(loss_weights=train_loss_weight)
            
        if not cur_round is None:
            print('***************current runing is based on %d round, this run will has %d epochs.****************' % (cur_round, epochs))

        '''
        if 'active_training' in self.config: 
            self.model_manager.fit_active_training(x_train, y_train, epochs, batch_size)

        elif 'batch_generator' in self.config:
        '''
        if 'batch_generator' in self.config:
            print('batch_generator:', self.config['batch_generator'])
            batch_generator_config = self.config['batch_generator'] 
            if not self.sg:
                batch_generator_config['batch_size'] = batch_size
                batch_generator_config['epochs'] = epochs
                context = {}
                context['data_manager'] = self.data_manager
                context['model_manager'] = self.model_manager
                self.sg = batch_generator_creator.instance.create(batch_generator_config, context)
            self.model_manager.fit(self.sg, epochs = epochs, validation_data=validation_data)
        elif 'training_method' in self.config:
            training_method_config = self.config['training_method']
            training_method_config['batch_size'] = batch_size
            training_method_config['epochs'] = epochs
            context = {}
            context['data_manager'] = self.data_manager
            context['model_manager'] = self.model_manager
            training_method = training_method_creator.instance.create(training_method_config, context)
            training_method.train()
        else:
        
            print('batch_generator:', 'default')
            
            self.model_manager.fit(x_train=x_train, y_train=y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

        suffix = ''
        if not cur_round is None:
            suffix += '_round_' + str(cur_round)

        self.model_manager.save_model(suffix)
            
        y_pred = self.model_manager.predict(x_test)

        for evaluator in self.evaluators:
            evaluator.evaluate(y_pred, y_test, len(x_test), self.config['print_report'])
        return 

def get_model_manager(self):
    return self.model_manager

def create(config, data_manager, model_manager, evaluator_list):
    return common_evaluator_manager(config, data_manager, model_manager, evaluator_list)

evaluator_manager_creator.instance.register(common_evaluator_manager.name, create)
