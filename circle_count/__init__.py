import keras

if keras.backend.backend() == "tensorflow":
    from backend.tensorflow.circle_count import *
