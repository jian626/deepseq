from framework import register

class creator(register.base):
    def create(self, config):
        name = config['name']
        fn = self.get_entry(name)
        return fn(config)

instance = creator()
