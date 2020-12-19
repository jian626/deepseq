from tensorflow.keras.utils import Sequence
class SequenceGenerator(Sequence):
    def __init__(self, data_manager, batch_size):
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.pos = 0
        x, y = self.data_manager.get_training_data()
        self.len = int(x.shape[0] / batch_size)

    def __getitem__(self):
        print('----------------------__getitem__-----------')
        x, y = self.data_manager.get_training_data()
        pos = self.pos
        self.pos += 1
        return x[pos: pos + batch_size], y[pos: pos + batch_size]

    def __len__(self):
        print('-----------------------__len__-------------:', self.len)
        return self.len

    def on_epoch_end(self):
        print('---------------------on_epoch_end------------')
