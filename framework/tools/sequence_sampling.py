from tensorflow.keras.utils import Sequence
class SequenceGenerator(Sequence):
    def __init__(self, data_manager, batch_size):
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.pos = 0
        x, y = self.data_manager.get_training_data()
        self.len = int(x.shape[0] / batch_size)

    def __getitem__(self, index):
        print('----------------------__getitem__-----------:', index)
        x, y = self.data_manager.get_training_data()
        pos = self.pos
        self.pos += 1
        rx = x[index: index + self.batch_size] 
        ry = []
        for i in range(4):
            ry.append(y[index][index:index+self.batch_size])
        return rx, ry

    def __len__(self):
        print('-----------------------__len__-------------:', self.len)
        return self.len

    def on_epoch_end(self):
        print('---------------------on_epoch_end------------')
