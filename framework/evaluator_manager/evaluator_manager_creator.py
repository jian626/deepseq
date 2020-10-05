from framework import register

class creator(register.base):
    def create(self, config, data_manager, model_manager, estimator_list):
        name = config['name']
        fn = self.get_entry(name)
        return fn(config, data_manager, model_manager, estimator_list)

instance = creator()
