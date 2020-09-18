import model_manager
import numpy as np


mc = model_manager.model_creator.load_model('./models/my_model')
y_pred = mc.predict_on_file('./uniprot-reviewed_yes.tab')
data_manager = mc.get_data_manager()
bool_labels = []

task_num = data_manager.get_task_num()
if task_num == 1:
    y_pred = [y_pred]

for i in range(data_manager.get_task_num()):
    bool_labels.append(y_pred[i] > 0.5)

labels = data_manager.one_hot_to_labels(bool_labels)
labels = labels[0]
f = open("result.txt", "w")
index = 0
for i in labels:
    if len(i) > 0:
        out = str(i[0]) + '\n'
    else:
        out = "None\n"
    f.writelines(out)
    index += 1
f.close()
    




