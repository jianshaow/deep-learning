import random

import numpy as np
from tensorflow import keras

import xor_util as util
from common import visualization as vis
from xor_util import SEQUENCE_SIZE, TRAINING_EPOCH


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(2, SEQUENCE_SIZE)),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(SEQUENCE_SIZE, activation=keras.activations.sigmoid)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy()])
    return model


if __name__ == '__main__':
    training_seq_pairs, training_labels = util.load_training_data()
    test_seq_pairs, test_labels = util.load_test_data()

    callback = vis.VisualizationCallback(show_model=True, runtime_plot=True)

    model = create_model()
    history = model.fit(training_seq_pairs,
                        training_labels,
                        validation_data=(test_seq_pairs, test_labels),
                        epochs=TRAINING_EPOCH,
                        callbacks=[callback]
                        )

    example_data = util.random_seq_pairs(1)
    example_result = model.predict(example_data)
    vis.build_multi_bar_figure(['seq1', 'seq2', 'xor'],
                               [example_data[0][0], example_data[0][1], example_result[0]])
    vis.show_all()
