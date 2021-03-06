from sklearn.metrics import classification_report
from framework import utili
from framework.strategy import hierarchical_learning
from framework.evaluator import evaluator_creator

class evaluator:
    name = 'enzyme_evaluator'
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def get_evaluate_report(self, pred_labels, index):
        task_num = self.data_manager.get_task_num()
        tested_classes = []
        conflict = [] 
        temp_msg = ''
        if task_num > 1:
            for i in range(task_num-1): 
                temp_conflict = hierarchical_learning.get_conflict(pred_labels[i+1][index], pred_labels[i][index], i+1)
                if temp_conflict:
                    temp_list = list(temp_conflict) 
                    conflict.append((i+1, i+2, temp_list))

        for i in range(task_num):
            for c in pred_labels[i][index]:
                 tested_classes.append(c)
        tested_classes = set(tested_classes)

        stat = []
        for c in tested_classes:
            cnt, level = self.data_manager.get_class_statistic(c)
            stat.append((c, cnt, level)) 
        sorted(stat, reverse=True, key=lambda e:e[2])
        return conflict, stat

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
                print('report level %d' % i)
                print(report)
                res = utili.strict_compare_report(target, pred, length)
                print('strict accuracy is %d of %d, %f%%' % (res, length, float(res) * 100.0 / length))

            res = {
                0:0, 
                1:0,
                2:0,
            }

        for i in range(length):
            for j in range(task_num-1):
                if hierarchical_learning.get_conflict(pred_labels[j+1][i], pred_labels[j][i], j+1):
                    res[j] += 1
        for i in range(task_num-1):
            print('comflict between level %d and level %d is %d, %f%% of %d.' % (i+1, i+2, res[i], float(res[i]) * 100.0 /float(length), length))
    
    
def create(data_manager):
    return evaluator(data_manager)

evaluator_creator.instance.register(evaluator.name, create)

    
    
