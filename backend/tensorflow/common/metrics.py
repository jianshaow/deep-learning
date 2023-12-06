import tensorflow as tf
import keras

import losses


class MeanMetricWrapper(keras.metrics.Mean):
    def __init__(self, fn, name=None):
        super(MeanMetricWrapper, self).__init__(name=name)
        self.fn = fn

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        matches = self.fn(y_true, y_pred)
        return super(MeanMetricWrapper, self).update_state(matches)

    def get_config(self):
        return self.fn.get_config()


def get_metric(metric, loss):
    if metric is None or isinstance(metric, keras.metrics.Mean):
        return metric

    if metric not in ["accuracy", "acc", "crossentropy", "ce"]:
        return MeanMetricWrapper(keras.metrics.get(metric), metric)

    is_sparse_categorical_crossentropy = isinstance(
        loss, keras.losses.SparseCategoricalCrossentropy
    ) or (
        isinstance(loss, losses.LossFunctionWrapper)
        and loss.fn == keras.losses.sparse_categorical_crossentropy
    )

    is_binary_crossentropy = isinstance(loss, keras.losses.BinaryCrossentropy) or (
        isinstance(loss, losses.LossFunctionWrapper)
        and loss.fn == keras.losses.binary_crossentropy
    )

    if metric in ["accuracy", "acc"]:
        if is_binary_crossentropy:
            return MeanMetricWrapper(keras.metrics.binary_accuracy, metric)
        elif is_sparse_categorical_crossentropy:
            return MeanMetricWrapper(keras.metrics.sparse_categorical_accuracy, metric)

    return MeanMetricWrapper(keras.metrics.categorical_accuracy, metric)
