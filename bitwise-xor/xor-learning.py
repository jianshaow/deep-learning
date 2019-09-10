import random

import numpy as np
from tensorflow import keras

from common import vis

SEQUENCE_SIZE = 10
TRAINING_DATA_SIZE = 5000
TEST_DATA_SIZE = 500
TRAINING_EPOCH = 10


def seq_xor(sequence_pair):
    xors = np.zeros((sequence_pair.shape[1]), dtype=np.uint8)

    for i in range(len(sequence_pair[0])):
        xors[i] = (sequence_pair[0][i] ^ sequence_pair[1][i])
    return xors


def random_seq():
    sequence = np.zeros((SEQUENCE_SIZE), dtype=np.uint8)
    for i in range(SEQUENCE_SIZE):
        sequence[i] = random.randint(0, 1)
    return sequence


def random_seq_pairs(size=10):
    seq_pairs = np.zeros((size, 2, SEQUENCE_SIZE), dtype=np.uint8)
    for i in range(size):
        seq_pairs[i][0] = random_seq()
        seq_pairs[i][1] = random_seq()
    return seq_pairs


def batch_xor(data):
    result = np.zeros((data.shape[0], SEQUENCE_SIZE), dtype=np.uint8)
    for i in range(len(data)):
        result[i] = seq_xor(data[i])
    return result


def load_training_date():
    training_seq_pairs = random_seq_pairs(TRAINING_DATA_SIZE)
    training_labels = batch_xor(training_seq_pairs)
    return training_seq_pairs, training_labels


def load_test_date():
    test_seq_pairs = random_seq_pairs(TEST_DATA_SIZE)
    test_labels = batch_xor(test_seq_pairs)
    return test_seq_pairs, test_labels


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
    training_seq_pairs, training_labels = load_training_date()
    test_seq_pairs, test_labels = load_test_date()

    callback = vis.VisualizationCallback(show_model=True, runtime_plot=True)

    model = create_model()
    history = model.fit(training_seq_pairs,
                        training_labels,
                        validation_data=(test_seq_pairs, test_labels),
                        epochs=TRAINING_EPOCH,
                        callbacks=[callback]
                        )

    example_data = random_seq_pairs(1)
    example_result = model.predict(example_data)
    vis.build_multi_bar_figure(['seq1', 'seq2', 'xor'],
                               [example_data[0][0], example_data[0][1], example_result[0]])
    vis.show_figures()
