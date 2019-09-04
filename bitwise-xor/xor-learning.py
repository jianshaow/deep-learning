import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

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


def show_history(history):
    plt.figure(figsize=(9, 6))

    accuracy = history.history['binary_accuracy']
    val_accuracy = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(accuracy, '.b-')
    plt.plot(val_accuracy, '.b--')
    last_index = len(accuracy) - 1
    plt.annotate('accuracy: %f' % accuracy[last_index],
                 xy=(last_index, accuracy[last_index]),
                 xytext=(-30, -50), textcoords='offset points', ha='center',
                 arrowprops=dict(facecolor='black', arrowstyle='-|>'))
    plt.annotate('val_accuracy: %f' % val_accuracy[last_index],
                 xy=(last_index, val_accuracy[last_index]),
                 xytext=(-90, -30), textcoords='offset points', ha='center',
                 arrowprops=dict(facecolor='black', arrowstyle='-|>'))

    plt.plot(loss, '.r-')
    plt.plot(val_loss, '.r--')
    last_index = len(loss) - 1
    plt.annotate('loss: %f' % loss[last_index],
                 xy=(last_index, loss[last_index]),
                 xytext=(-20, 50), textcoords='offset points', ha='center',
                 arrowprops=dict(facecolor='black', arrowstyle='-|>'))
    plt.annotate('val_loss: %f' % val_loss[last_index],
                 xy=(last_index, val_loss[last_index]),
                 xytext=(-80, 30), textcoords='offset points', ha='center',
                 arrowprops=dict(facecolor='black', arrowstyle='-|>'))

    plt.title('Training Loss & Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.xticks(range(TRAINING_EPOCH), range(1, TRAINING_EPOCH + 1))
    plt.legend(['accuracy', 'val_accuracy', 'loss',
                'val_loss'], loc='center right')

    plt.show()


def show_example(data, example):
    plt.figure(figsize=(9, 6))

    plt.subplot(3, 1, 1)
    plt.xticks(range(SEQUENCE_SIZE), range(1, SEQUENCE_SIZE + 1))
    plt.yticks([1])
    plt.ylabel('seq 1')
    plt.bar(range(SEQUENCE_SIZE), data[0], width=1, edgecolor='black')

    plt.subplot(3, 1, 2)
    plt.xticks(range(SEQUENCE_SIZE), range(1, SEQUENCE_SIZE + 1))
    plt.yticks([1])
    plt.ylabel('seq 2')
    plt.bar(range(SEQUENCE_SIZE), data[1], width=1, edgecolor='black')

    plt.subplot(3, 1, 3)
    plt.xticks(range(SEQUENCE_SIZE), range(1, SEQUENCE_SIZE + 1))
    plt.yticks([1])
    plt.ylabel('xor')
    plt.bar(range(SEQUENCE_SIZE), example, width=1, edgecolor='black')

    plt.show()


if __name__ == '__main__':
    training_seq_pairs, training_labels = load_training_date()
    test_seq_pairs, test_labels = load_test_date()

    model = create_model()
    history = model.fit(training_seq_pairs,
                        training_labels,
                        validation_data=(test_seq_pairs, test_labels),
                        epochs=TRAINING_EPOCH)
    show_history(history)

    example_data = random_seq_pairs(1)
    example_result = model.predict(example_data)
    show_example(example_data[0], example_result[0])
