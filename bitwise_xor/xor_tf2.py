import numpy as np
import tensorflow as tf
from tensorflow import keras

import xor_util as util
from xor_util import SEQUENCE_SIZE, TRAINING_EPOCH


class XORModel(tf.keras.Model):
    def __init__(self):
        super(XORModel, self).__init__()

        self.flatten_layer = keras.layers.Flatten()
        self.dense_layer1 = keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_layer2 = keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_layer3 = keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.output_layer = tf.compat.v1.layers.Dense(
            units=SEQUENCE_SIZE, activation=tf.nn.sigmoid)

    def call(self, data):
        x = self.flatten_layer(data)
        y = self.dense_layer1(x)
        y = self.dense_layer2(y)
        y = self.dense_layer3(y)
        return self.output_layer(y)


loss_object = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam()


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
        pred = model(inputs)
        loss_value = loss_object(targets, pred)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def loss(model, x, y):
    y_ = model(x)

    return loss_object(y_true=y, y_pred=y_)


def train():
    dataset = prepare_data()

    model = XORModel()

    for data, labels in dataset:
        loss_value, grads = grad(model, data, labels)
        print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                                  loss_value.numpy()))

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("kernel={}, bias={}".format(
        model.output_layer.kernel, model.output_layer.bias))

    # do_train(iterator, step, loss)


def prepare_data():
    training_data, training_labels = util.load_training_data()
    # test_data, test_labels = util.load_training_data()

    dataset = tf.data.Dataset.from_tensor_slices(
        (training_data, training_labels))
    dataset = dataset.batch(1).take(1)

    return dataset


if __name__ == '__main__':
    train()
