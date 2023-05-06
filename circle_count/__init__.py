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
DATA_NAME_PREFIX = 'circle_count'
DATA_SET_PATH = os.path.join(data_dir, 'dataset')
LEARNING_RATE = 0.00001


def data_config(
    r_lower, r_upper=None, circles_max=DEFAULT_CIRCLES_MAX, error_of='error'
):
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
    config['error_path'] = os.path.join(
        DATA_SET_PATH, data_name + '.' + error_of + '.npz'
    )

    def get_config(key='radius'):
        if key == 'radius':
            if r_lower and r_upper:
                return random.randint(r_lower, r_upper)
            else:
                return r_lower
        return config[key]

    return get_config


DEFAULT_DATA_CONFIG = data_config(6, 8)
