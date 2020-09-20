from framework.estimator_manager import estimator_manager
def create(config, data_manager, model_manager, estimator_list):
    if 'name' in config:
        if config['name'] == 'common_estimator_manager':
            return estimator_manager.common_estimator_manager(config, data_manager, model_manager, estimator_list)
        else:
            Exception('unsupported:' % config['name'])
    return estimator_manager.common_estimator_manager(config, data_manager, model_manager, estimator_list)
