from tensorflow.keras.utils import Sequence
class SequenceGenerator(Sequence):
    def __init__(self, data_manager, batch_size):
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.pos = 0
        x, y = self.data_manager.get_training_data()
        self.len = int(x.shape[0] / batch_size)

    def __getitem__(self, index):
        print('----------------------__getitem__-----------')
        x, y = self.data_manager.get_training_data()
        pos = self.pos
        self.pos += 1
        rx, ry = x[pos: pos + self.batch_size], y[pos: pos + self.batch_size]
        print('len y:', len(y)
        print('len rx:', len(rx))
        print('len ry:', len(ry))
        return rx, ry

    def __len__(self):
        print('-----------------------__len__-------------:', self.len)
        return self.len

    def on_epoch_end(self):
        print('---------------------on_epoch_end------------')
