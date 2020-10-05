
from sklearn.metrics import classification_report
from framework import utili
from framework.bio import process_enzyme
from framework.evaluator import evaluator_creator

class evaluator:
    name = 'enzyme_protein_evaluator'
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def evaluate(self, y_pred, y_test, length, print_report):
        task_num = self.data_manager.get_task_num()
        if task_num == 1:
            y_pred = [y_pred]

        map_table = {} 

        bool_labels = []
        for i in range(task_num):
            bool_labels.append(y_pred[i] > 0.5)

        pred_labels = self.data_manager.one_hot_to_labels(bool_labels)

        for i in range(task_num):
            pred = bool_labels[i] 
            target = y_test[i]
            report = classification_report(target, pred)

            if print_report:
                print(report)
    
    
    

def create(data_manager):
    return evaluator(data_manager)

evaluator_creator.instance.register(evaluator.name, create)
    
