import tensorflow as tf

class SimepleFlatten():
    def __init__(self, units, input_shape=None):
        self.units = units
        self.input_shape = input_shape

    def __call__(self, inputs):
        return inputs

    @property
    def trainable_variables(self):
        pass

class SimepleDense():
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.kernel = None
        self.bias = None

    def __call__(self, inputs):
        self.kernel = tf.Variable()
        self.bias = tf.Variable()

        y = inputs * self.kernel + self.bias
        if self.activation:
            y = self.activation(y)
        return y

    @property
    def trainable_variables(self):
        pass
