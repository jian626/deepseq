from framework.data_manager import enzyme_data_manager
from framework.data_manager import enzyme_protein_data_manager

def create(config):
    if 'enzyme_data_manager' == config['name']:
        return enzyme_data_manager.enzyme_data_manager(config)
    elif 'enzyme_protein_data_manager' == config['name']:
        return enzyme_protein_data_manager.enzyme_protein_data_manager(config)
        
