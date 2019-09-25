import tensorflow as tf
from tensorflow import keras
from common import callbacks as cbs


class SimpleModel():

    def __init__(self, layers):
        self.name = 'Simple Model'
        self._is_graph_network = False
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

    @property
    def metrics_names(self):
        metrics_names = ['loss']
        metrics_names += [m.name for m in self._metrics]
        return metrics_names

    def __reset_metrics(self):
        self.loss_mean.reset_states()
        for metric in self._metrics:
            metric.reset_states()

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.loss_mean = keras.metrics.Mean('loss')
        self._metrics = metrics

    def fit(self, dataset, epochs, callbacks=[], verbose=True, validation_data=None):
        callbacks = [keras.callbacks.BaseLogger()] + \
            (callbacks or []) + [self.history]
        if verbose:
            callbacks.append(keras.callbacks.ProgbarLogger())
        callbacks = cbs.CallbackList(callbacks)

        callbacks.set_model(self)
        callback_metrics = list(self.metrics_names)
        if validation_data:
            callback_metrics += ['val_' + n for n in self.metrics_names]
        params = {'metrics': callback_metrics, 'epochs': epochs,
                  'samples': 5000, 'verbose': verbose}
        callbacks.set_params(params)

        callbacks.on_train_begin()

        for epoch in range(epochs):
            self.__reset_metrics()
            epoch_logs = dict()
            callbacks.on_epoch_begin(epoch)

            loss_value = None
            for batch, (data, labels) in dataset.enumerate():
                batch_value = batch.numpy()
                callbacks.on_batch_begin(batch_value)
                with tf.GradientTape() as tape:
                    preds = self(data)
                    loss_value = self.loss(labels, preds)
                    grads = tape.gradient(
                        loss_value, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.trainable_variables))

                batch_logs = {'size': len(
                    data), 'batch': batch_value, 'loss': loss_value.numpy()}
                self.loss_mean(loss_value)
                for metric in self._metrics:
                    metric_value = metric(labels, preds)
                    batch_logs[metric.name] = metric_value.numpy()

                callbacks.on_batch_end(batch_value, batch_logs)

            epoch_logs['loss'] = self.loss_mean.result().numpy()
            for metric in self._metrics:
                epoch_logs[metric.name] = metric.result().numpy()

            if validation_data:
                self.__reset_metrics()
                (test_data, test_labels) = validation_data
                test_preds = self(test_data)
                test_loss_value = self.loss(test_labels, test_preds)
                epoch_logs['val_loss'] = test_loss_value.numpy()
                for metric in self._metrics:
                    test_metric_value = metric(test_labels, test_preds)
                    epoch_logs['val_' +
                               metric.name] = test_metric_value.numpy()

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end()

        return self.history

    def predict(self, data):
        return self(data).numpy()

    def __metrics_log(self):
        log = '- '
        for metric in self._metrics:
            log = log + metric.name + ': ' + str(metric.result().numpy())
        return log


class History(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
