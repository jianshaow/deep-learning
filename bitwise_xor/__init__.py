import keras

if keras.backend.backend() == "tensorflow":
    import backend.tensorflow.bitwise_xor.xor_tf2_layer as xor_tf2_layer
    import backend.tensorflow.bitwise_xor.xor_tf2_model as xor_tf2_model

if keras.backend.backend() == "torch":
    import backend.pytorch.bitwise_xor.xor_torch as xor_torch
