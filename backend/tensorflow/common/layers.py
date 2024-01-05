import numpy as np
import tensorflow as tf
import keras


class Layer:
    name_prefix = "layer"

    def __init__(self, name=None, dtype=tf.float32):
        if name is None:
            self.name = gen_name(self.__class__.name_prefix)
        else:
            self.name = name
        self.dtype = dtype
        self._inbound_nodes = []

    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        return inputs


class SimpleInput(Layer):
    name_prefix = "input"

    def __init__(self, input_shape=None):
        super(SimpleInput, self).__init__()
        self.input_shape = input_shape


class SimpleFlatten(Layer):
    name_prefix = "flatten"

    def __init__(self, units=None, input_shape=None, dtype=tf.float32):
        super(SimpleFlatten, self).__init__(dtype=dtype)
        self.name = ""
        self.units = units
        self.input_shape = input_shape
        self._batch_input_shape = (None,) + input_shape

    @property
    def trainable_variables(self):
        return {}

    def call(self, inputs):
        input_shape = inputs.shape[1:]
        if self.input_shape and self.input_shape != input_shape:
            raise ValueError(
                "expected input shape "
                + str(self.input_shape)
                + ", but got input with shape "
                + str(input_shape)
            )
        return tf.reshape(inputs, (-1, np.prod(input_shape, dtype=int)))


class SimpleDense(Layer):
    name_prefix = "dense"

    def __init__(
        self,
        units,
        activation=None,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Zeros(),
        **kwargs
    ):
        super(SimpleDense, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.weights = []
        self.kernel = None
        self.bias = None
        self._built = False

    @property
    def trainable_variables(self):
        return self.weights

    def call(self, inputs):
        if not inputs.dtype == tf.float32:
            inputs = tf.cast(inputs, dtype=tf.float32)

        self._maybe_build(inputs.shape)

        y = tf.matmul(inputs, self.kernel)
        y = tf.nn.bias_add(y, self.bias)

        if self.activation:
            y = self.activation(y)
        return y

    def _maybe_build(self, input_shape):
        if not self._built:
            self.kernel = self.add_weight(
                "kernel", (input_shape[-1], self.units), self.kernel_initializer
            )
            self.bias = self.add_weight("bias", (self.units), self.bias_initializer)
            self._built = True

    def add_weight(self, name, shape=None, initializer=None):
        weight = tf.Variable(initializer(shape), name=name, shape=shape)
        self.weights.append(weight)
        return weight


name_indexes = {}


def gen_name(name_prefix):
    index = name_indexes.get(name_prefix)
    if index is None:
        name_indexes[name_prefix] = 0
        return name_prefix
    else:
        index += 1
        name_indexes[name_prefix] = index
        return name_prefix + "_" + str(index)


if __name__ == "__main__":
    layer = Layer()
    print(layer.name)
    layer = Layer()
    print(layer.name)
    layer = SimpleFlatten(1)
    print(layer.name)
    layer = SimpleFlatten(1)
    print(layer.name)
    layer = SimpleDense(1)
    print(layer.name)
    layer = SimpleDense(1)
    print(layer.name)
