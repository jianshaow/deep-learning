import tensorflow as tf
from tensorflow import keras


class SimpleModel():

    def __init__(self, layers):
        self.name = 'Simple Model'
        self._is_graph_network = True
        self._layers = layers
        self.history = dict()

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
        for callback in callbacks:
            callback.set_model(self)
            params = {'metrics': self.__get_metrics(), 'epochs': epochs}
            callback.set_params(params)
            callback.on_train_begin()

        for epoch in range(epochs):
            self.__reset_metrics()
            logs = dict()
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
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
            self.__print_log(epoch, epochs)

        for callback in callbacks:
            callback.on_train_end()

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
