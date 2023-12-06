import keras

if keras.backend.backend() == "tensorflow":
    import backend.tensorflow.mnist.cls_tf2_layer as cls_tf2_layer
