from tensorflow.keras.utils import Sequence
import nump as np
class SequenceGenerator(Sequence):
    def __init__(self, data_manager, batch_size):
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.pos = 0
        x, y = self.data_manager.get_training_data()
        self.len = int(np.floor(len(x) / self.batch_size))

    def __getitem__(self, index):
        print('----------------------__getitem__-----------:', index)
        x, y = self.data_manager.get_training_data()
        task_num = self.data_manager.get_task_num()
        pos = self.pos
        self.pos += 1
        rx = x[index: index + self.batch_size] 
        ry = []
        for i in range(task_num):
            ry.append(y[i][index:index+self.batch_size])
        return rx, ry

    def __len__(self):
        print('-----------------------__len__-------------:', self.len)
        return self.len

    def on_epoch_end(self):
        print('---------------------on_epoch_end------------')
