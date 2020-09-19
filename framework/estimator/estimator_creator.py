from framework.estimator import enzyme_protein_estimator
from framework.estimator import enzyme_estimator


def create(name, data_manager):
    if name == 'enzyme_estimator':
        return enzyme_estimator.estimator(data_manager)
    elif name == 'enzyme_protein_estimator':
        return enzyme_protein_estimator.estimator(data_manager)
    raise Exception('unsupported extimator name:%s' % name)
