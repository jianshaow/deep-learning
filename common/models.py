import tensorflow as tf
from tensorflow import keras

from common import callbacks as cbs
from common import layers, losses
from common import metrics as mtx


class SimpleSequential(layers.Layer):

    def __init__(self, layers):
        super(SimpleSequential, self).__init__()
        self._is_graph_network = False
        self._layers = []
        self.history = History()

        for layer in layers:
            self.add(layer)

    def add(self, layer):
        self._layers.append(layer)

    def call(self, data):
        input = data
        output = input
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

    def reset_metrics(self):
        self.loss_mean.reset_states()
        for metric in self._metrics:
            metric.reset_states()

    def compile(self, optimizer, loss, metrics):
        self.optimizer = keras.optimizers.get(optimizer)
        self.loss = losses.get_loss(loss)
        self.loss_mean = keras.metrics.Mean()
        self._metrics = [mtx.get_metric(metric, self.loss)
                         for metric in metrics]

    def fit(self, dataset, epochs, callbacks=[], verbose=True, validation_data=None):
        callbacks = [keras.callbacks.BaseLogger()] + \
            (callbacks or []) + [self.history]
        if verbose:
            callbacks.append(keras.callbacks.ProgbarLogger(count_mode='steps'))
        callbacks = cbs.CallbackList(callbacks)

        callbacks.set_model(self)
        callback_metrics = list(self.metrics_names)
        if validation_data:
            callback_metrics += ['val_' + n for n in self.metrics_names]
        steps = tf.data.experimental.cardinality(dataset).numpy()
        params = {'metrics': callback_metrics, 'epochs': epochs,
                  'steps': steps, 'verbose': verbose}
        callbacks.set_params(params)

        callbacks.on_train_begin()

        for epoch in range(epochs):
            self.reset_metrics()
            epoch_logs = {}
            callbacks.on_epoch_begin(epoch)

            for batch, train_data in dataset.enumerate():
                batch_value = batch.numpy()
                callbacks.on_batch_begin(batch_value)
                batch_logs = {'size': len(train_data), 'batch': batch_value}
                self.train_step(train_data, batch_logs)
                callbacks.on_batch_end(batch_value, batch_logs)

            epoch_logs['loss'] = self.loss_mean.result().numpy()
            for metric in self._metrics:
                epoch_logs[metric.name] = metric.result().numpy()

            if validation_data:
                self.test_step(validation_data, epoch_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end()

        return self.history

    def train_step(self, train_data, batch_logs):
        (data, labels) = train_data
        with tf.GradientTape() as tape:
            preds = self(data)
            loss_value = self.loss(labels, preds)
            grads = tape.gradient(
                loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables))
        self.loss_mean(loss_value)
        batch_logs['loss'] = loss_value.numpy()

        for metric in self._metrics:
            metric_value = metric(labels, preds)
            batch_logs[metric.name] = metric_value.numpy()

    def test_step(self, validation_data, epoch_logs):
        (test_data, test_labels) = validation_data
        self.reset_metrics()
        test_preds = self(test_data)
        test_loss_value = self.loss(test_labels, test_preds)
        epoch_logs['val_loss'] = test_loss_value.numpy()
        for metric in self._metrics:
            test_metric_value = metric(test_labels, test_preds)
            epoch_logs['val_' +
                       metric.name] = test_metric_value.numpy()

    def predict(self, data):
        return self(data).numpy()


class History(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
