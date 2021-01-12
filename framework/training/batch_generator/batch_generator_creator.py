from framework import register

class creator(register.base):
    def create(self, config, data_manager):
        name = config['name']
        fn = self.get_entry(name)
        return fn(config, data_manager)

instance = creator()

