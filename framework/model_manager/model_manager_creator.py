from framework.model_manager import model_manager
from framework import utili
from framework.data_manager import data_manager_creator
from tensorflow.keras.models import load_model

def create(data_manager, config):
    if 'name' in config:
        if config['name'] == 'model_common_manager':
            return model_manager.model_common_manager(data_manager, config)
    else:
        ''' default
        '''
        return model_manager.model_common_manager(data_manager, config)

def create_from_file(name):
    model_name = name + '.h5'
    store_name = name 

    store = utili.load_obj(store_name)
    data_manager = data_manager_creator.create(store['data_manager_info'])
    model_manager = create(data_manager, store['config'])
    model = load_model(model_name)
    model_manager.set_model(model)
    return model_manager
