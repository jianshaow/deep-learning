import os
import random
from common import data_dir

import keras

if keras.backend.backend() == "tensorflow":
    from backend.tensorflow.circle_count import dataset


REG_MODEL_PARAMS = {
    "model_type": "RegressionModel",
    "input_shape": (100, 100, 1),
    "fc_layers": 3,
    "fc_layers_units": 64,
}

CLS_MODEL_PARAMS = {
    "model_type": "ClassificationModel",
    "input_shape": (100, 100, 1),
    "fc_layers": 3,
    "fc_layers_units": 64,
    "output_units": 6,
}

CONV_REG_MODEL_PARAMS = {
    "model_type": "ConvRegModel",
    "input_shape": (100, 100, 1),
    "conv_layers": 2,
    "conv_filters": 64,
    "fc_layers": 1,
    "fc_layers_units": 128,
}

CONV_CLS_MODEL_PARAMS = {
    "model_type": "ConvClsModel",
    "input_shape": (100, 100, 1),
    "conv_layers": 1,
    "fc_layers": 2,
    "conv_filters": 64,
    "fc_layers_units": 32,
    "output_units": 6,
}

DEFAULT_RADIUS_RANGE = (6, 8)
DEFAULT_CIRCLES_RANGE = (0, 5)
DATA_NAME_PREFIX = "circles"
DATA_SET_PATH = os.path.join(data_dir, "dataset")
LEARNING_RATE = 0.00001


def data_config(
    r_range=DEFAULT_RADIUS_RANGE,
    q_range=DEFAULT_CIRCLES_RANGE,
):
    r_lower, r_upper = r_range
    q_lower, q_upper = q_range

    config = __base_data_config(r_lower, r_upper, q_lower, q_upper)

    def get_config(key, **kwargs):
        if key == "radius_fn":

            def r_fn():
                if r_upper is not None:
                    return random.randint(r_lower, r_upper)
                else:
                    return r_lower

            return r_fn
        elif key == "quantity_fn":

            def q_fn():
                if q_upper is not None:
                    return random.randint(q_lower, q_upper)
                else:
                    return q_lower

            return q_fn
        elif key == "error_path":
            if kwargs.get("error_gt"):
                error_gt = kwargs["error_gt"]
            else:
                error_gt = 0
            error_data_file = "%s.error_gt-%s.npz" % (config["name"], error_gt)
            return os.path.join(DATA_SET_PATH, error_data_file)

        return config[key]

    return get_config


def __base_data_config(r_lower, r_upper, q_lower, q_upper):
    config = {}
    config["r_lower"] = r_lower
    config["q_lower"] = q_lower

    data_name = "%s.r%02d" % (DATA_NAME_PREFIX, r_lower)
    if r_upper is not None:
        config["r_upper"] = r_upper
        data_name = "%s-%02d" % (data_name, r_upper)

    data_name = "%s.q%02d" % (data_name, q_lower)
    if q_upper is not None:
        config["q_upper"] = q_upper
        data_name = "%s-%02d" % (data_name, q_upper)

    config["name"] = data_name
    config["path"] = os.path.join(DATA_SET_PATH, data_name + ".npz")

    return config


DEFAULT_DATA_CONFIG = data_config()
