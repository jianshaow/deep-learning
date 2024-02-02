import keras

if keras.backend.backend() == "tensorflow":
    from backend.tensorflow.mnist_demo import cls_tf2_layer
    from backend.tensorflow.mnist_demo import dataset
if keras.backend.backend() == "torch":
    from backend.pytorch.mnist_demo import dataset
