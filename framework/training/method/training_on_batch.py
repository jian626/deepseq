from framework.training.method import training_method_creator as creator

class training_on_batch:
    name = 'training_on_batch'
    def __init__(self, config, context):
        self.config = config
        self.context = context

    def train(self):
        print('--------------------------training------------------')


def create(config, context):
    return training_on_batch(config, context)

creator.instance.register(training_on_batch.name, create)
