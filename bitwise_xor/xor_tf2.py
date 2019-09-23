import numpy as np
import tensorflow as tf
from tensorflow import keras

import xor_util as util
from common import visualization as vis
from xor_util import SEQUENCE_SIZE, TRAINING_EPOCH


class XORModel(tf.keras.Model):
    def __init__(self):
        super(XORModel, self).__init__()

        self.flatten_layer = keras.layers.Flatten()
        self.dense_layer1 = keras.layers.Dense(
            units=64, activation=keras.activations.relu)
        self.dense_layer2 = keras.layers.Dense(
            units=64, activation=keras.activations.relu)
        self.dense_layer3 = keras.layers.Dense(
            units=64, activation=keras.activations.relu)
        self.output_layer = keras.layers.Dense(
            units=SEQUENCE_SIZE, activation=keras.activations.sigmoid)

    def call(self, data):
        x = self.flatten_layer(data)
        y = self.dense_layer1(x)
        y = self.dense_layer2(y)
        y = self.dense_layer3(y)
        return self.output_layer(y)


def do_train(dataset, model):
    loss_object = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(0.001)

    epochs = []
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(TRAINING_EPOCH):
        epoch_loss_avg = keras.metrics.Mean()
        epoch_accuracy = keras.metrics.BinaryAccuracy()

        for data, labels in dataset:
            with tf.GradientTape() as tape:
                preds = model(data)
                loss_value = loss_object(labels, preds)
                grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg(loss_value)
            epoch_accuracy(labels, preds)

        epochs.append(epoch)
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        print('epoch {}: loss={}, accuracy={}'.format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()))


def prepare_data():
    training_data, training_labels = util.load_training_data()
    # test_data, test_labels = util.load_training_data()

    dataset = tf.data.Dataset.from_tensor_slices(
        (training_data, training_labels))
    dataset = dataset.batch(32)  # .take(10)

    return dataset


if __name__ == '__main__':
    dataset = prepare_data()

    model = XORModel()

    # model.compile(optimizer=keras.optimizers.Adam(),
    #             loss=keras.losses.BinaryCrossentropy(),
    #             metrics=[keras.metrics.BinaryAccuracy()])

    # callback = vis.VisualizationCallback(show_model=True, runtime_plot=True)
    # model.fit(dataset, epochs=10, callbacks=[callback])

    do_train(dataset, model)

    model.summary()
    # print(model.trainable_variables)
