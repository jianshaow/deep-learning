import keras

if keras.backend.backend() == "tensorflow":
    import backend.tensorflow.mnist.cls_tf2_layer as cls_tf2_layer
if keras.backend.backend() == "torch":
    from backend.pytorch.mnist import dataset
