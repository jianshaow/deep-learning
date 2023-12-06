import keras


class LossFunctionWrapper(keras.losses.Loss):
    def __init__(self, fn, name=None):
        super(LossFunctionWrapper, self).__init__(name=name)
        self.fn = fn

    def call(self, y_true, y_pred):
        return self.fn(y_true, y_pred)


def get_loss(loss):
    if loss is None or isinstance(loss, keras.losses.Loss):
        return loss
    return LossFunctionWrapper(keras.losses.get(loss), loss)
