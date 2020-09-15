from sklearn.metrics import classification_report
import utili
import process_enzyme

class enzyme_estimator:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def estimate(self, y_pred, y_test, length, print_report):
        task_num = self.data_manager.get_task_num()
        if task_num == 1:
            y_pred = [y_pred]

        field_map_to_number = self.data_manager.get_feature_mapping()
        map_table = {} 
        class_res = {}

        for i in range(task_num):
            pred = (y_pred[i] > 0.5)
            target = y_test[i]
            report = classification_report(target, pred)
            map_table[i] = utili.switch_key_value(field_map_to_number[i])

            if print_report:
                print('report level %d' % i)
                print(report)
                res = utili.strict_compare_report(target, pred, length)
                print('strict accuracy is %d of %d, %f%%' % (res, length, float(res) * 100.0 / length))

            temp = []
            for y_ in pred:
                temp.append(utili.map_label_to_class(map_table[i], y_))
            class_res[i] = temp

            res = {
                0:0, 
                1:0,
                2:0,
            }

        for i in range(length):
            for j in range(task_num-1):
                if process_enzyme.is_conflict(class_res[j+1][i], class_res[j][i], j+1):
                    res[j] += 1
    
        for i in range(task_num-1):
            print('comflict between level %d and level %d is %d, %f%% of %d.' % (i+1, i+2, res[i], float(res[i]) * 100.0 /float(length), length))
    
    
    
