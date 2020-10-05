from framework import register
class creator(register.base):
    def create(self, name, data_manager):
        fn = self.get_entry(name)
        return fn(data_manager)


instance = creator()
