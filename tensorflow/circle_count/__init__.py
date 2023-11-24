import os
import random

from common import data_dir

DEFAULT_MODEL_PARAMS = {
    'input_shape': (100, 100),
    'hidden_layers': 3,
    'hidden_layer_units': 64,
    'output_units': 6,
}

DEFAULT_CIRCLES_MAX = 6
DATA_NAME_PREFIX = 'circles'
DATA_SET_PATH = os.path.join(data_dir, 'dataset')
LEARNING_RATE = 0.00001


def data_config(r_lower, r_upper=None, circles_max=DEFAULT_CIRCLES_MAX):
    config = __base_data_config(r_lower, r_upper, circles_max)

    def get_config(key='radius', **kwargs):
        if key== 'radius':
            if r_lower and r_upper:
                return random.randint(r_lower, r_upper)
            else:
                return r_lower
        elif key == 'error_path':
            if kwargs.get('error_gt'):
                error_gt = kwargs['error_gt']
            else:
                error_gt = 0
            error_data_file = '%s.error_gt-%s.npz' % (config['name'], error_gt)
            return os.path.join(DATA_SET_PATH, error_data_file)

        return config[key]

    return get_config


def __base_data_config(r_lower, r_upper, circles_max):
    config = {}
    config['r_lower'] = r_lower
    config['circles_max'] = circles_max

    data_name = '%s.%02d' % (DATA_NAME_PREFIX, r_lower)
    if r_upper is not None:
        config['r_upper'] = r_upper
        data_name = '%s-%02d' % (data_name, r_upper)
    data_name = '%s.%02d' % (data_name, circles_max)
    config['name'] = data_name
    config['path'] = os.path.join(DATA_SET_PATH, data_name + '.npz')

    return config


DEFAULT_DATA_CONFIG = data_config(6, 8)
