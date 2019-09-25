

class SimepleDense():
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.kernel = None
        self.bias = None

    def __call__(self, x):
        y = x * self.kernel + self.bias
        if self.activation:
            y = self.activation(y)
        return y

    @property
    def trainable_variables(self):
        pass
