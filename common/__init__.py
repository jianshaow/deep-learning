import keras

if keras.backend.backend() == "tensorflow":
    import backend.tensorflow.common.callbacks as callbacks
    import backend.tensorflow.common.layers as layers
    import backend.tensorflow.common.losses as losses
    import backend.tensorflow.common.metrics as metrics
    import backend.tensorflow.common.models as models

import os

data_dir = os.path.expanduser("~/.deep-learning")
