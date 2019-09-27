import numpy as np
import tensorflow as tf
from tensorflow import keras


class SimpleFlatten():
    def __init__(self,
                 units=None,
                 input_shape=None):
        self.units = units
        self.input_shape = input_shape

    @property
    def trainable_variables(self):
        return {}

    def __call__(self, inputs):
        input_shape = inputs.shape
        return tf.reshape(inputs, (-1, np.prod(input_shape[1:], dtype=int)))


class SimpleDense():
    def __init__(self,
                 units,
                 activation=None,
                 kernel_initializer=keras.initializers.GlorotUniform(),
                 bias_initializer=keras.initializers.Zeros()):
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.weights = []
        self.kernel = None
        self.bias = None
        self._built = False

    @property
    def trainable_variables(self):
        return self.weights

    def __call__(self, inputs):
        if not inputs.dtype.is_floating:
            inputs = tf.dtypes.cast(inputs, dtype=tf.float32)

        self._maybe_build(inputs.shape)

        y = tf.matmul(inputs, self.kernel)
        y = tf.nn.bias_add(y, self.bias)

        if self.activation:
            y = self.activation(y)
        return y

    def _maybe_build(self, input_shape):
        if not self._built:
            self.kernel = self.add_weight(
                'kernel', (input_shape[-1], self.units), self.kernel_initializer)
            self.bias = self.add_weight(
                'bias', (self.units), self.bias_initializer)
            self._built = True

    def add_weight(self, name, shape=None, initializer=None):
        weight = tf.Variable(initializer(shape), name=name, shape=shape)
        self.weights.append(weight)
        return weight
