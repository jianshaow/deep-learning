import tensorflow as tf
from tensorflow import keras


class SimpleModel():

    def __init__(self, layers):
        self.name = 'Simple Model'
        self._is_graph_network = True
        self._layers = layers
        self.history = History()

    def __call__(self, data):
        input = data
        for layer in self._layers:
            output = layer(input)
            input = output
        return output

    @property
    def trainable_variables(self):
        variables = []
        for layer in self._layers:
            variables.extend(layer.trainable_variables)
        return variables

    def __reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()
        self.loss_mean.reset_states()

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.loss_mean = keras.metrics.Mean()

    def fit(self, dataset, epochs, callbacks=[]):
        callbacks = CallbackList(callbacks)
        callbacks.append(keras.callbacks.BaseLogger())
        callbacks.append(self.history)
        callbacks.set_model(self)
        params = {'metrics': self.__get_metrics(), 'epochs': epochs}
        callbacks.set_params(params)

        callbacks.on_train_begin()

        for epoch in range(epochs):
            self.__reset_metrics()
            logs = dict()
            callbacks.on_epoch_begin(epoch, logs)
            for data, labels in dataset:
                with tf.GradientTape() as tape:
                    preds = self(data)
                    loss_value = self.loss(labels, preds)
                    grads = tape.gradient(
                        loss_value, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.trainable_variables))

                self.loss_mean(loss_value)
                for metric in self.metrics:
                    metric(labels, preds)

            logs['loss'] = self.loss_mean.result().numpy()
            for metric in self.metrics:
                logs[metric.name] = metric.result().numpy()
            callbacks.on_epoch_end(epoch, logs)
            self.__print_log(epoch, epochs)

        callbacks.on_train_end()

        return self.history

    def predict(self, data):
        return self(data)

    def __print_log(self, epoch, epochs):
        template = 'Epoch {}/{} - loss: {}'
        print(template.format(epoch+1, epochs, self.loss_mean.result()),
              self.__metrics_log())

    def __get_metrics(self):
        metric_names = ['loss']
        for metric in self.metrics:
            metric_names.append(metric.name)
        return metric_names

    def __metrics_log(self):
        log = '- '
        for metric in self.metrics:
            log = log + metric.name + ': ' + str(metric.result().numpy())
        return log


class CallbackList():
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.params = {}
        self.model = None

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)


class History(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
