
class base:  
    name = 'creator_base'
    def __init__(self):
        self.register_table = {}

    def register(self, name, obj):
        if name in self.register_table:
            raise Exception('the name, %s, has been used.' % name)
        self.set_entry(name, obj)

    def get_table(self):
        return self.register_table

    def get_entry(self, name):
        return self.register_table[name]

    def set_entry(self, name, obj):
        self.register_table[name] = obj
        
