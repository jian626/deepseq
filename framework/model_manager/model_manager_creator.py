from framework import utili
from framework.data_manager import data_manager_creator
from framework import register
from tensorflow.keras.models import load_model

class creator(register.base):
    def create(self, data_manager, config):
        name = config['name']
        fn = self.get_entry(name)
        return fn(data_manager, config)

    def create_from_file(self, name):
        model_name = name + '.h5'
        store_name = name 
    
        store = utili.load_obj(store_name)
        data_manager = data_manager_creator.instance.create(store['data_manager_info'])
        model_manager = self.create(data_manager, store['config'])
        model = load_model(model_name)
        model_manager.set_model(model)
        return model_manager


instance = creator()
