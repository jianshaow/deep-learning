import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

SEQUENCE_SIZE = 10
TRAINNING_DATA_SIZE = 5000
TEST_DATA_SIZE = 100
TRAINNING_EPOCH = 20


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


def load_trainning_date():
    trainning_seq_pairs = random_seq_pairs(TRAINNING_DATA_SIZE)
    trainning_labels = batch_xor(trainning_seq_pairs)
    return trainning_seq_pairs, trainning_labels


def load_test_date():
    test_seq_pairs = random_seq_pairs(TEST_DATA_SIZE)
    test_labels = batch_xor(test_seq_pairs)
    return test_seq_pairs, test_labels


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(2, SEQUENCE_SIZE)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(SEQUENCE_SIZE, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def show_history(history):
    plt.figure(figsize=(9, 6))

    accuracy = history.history['acc']
    loss = history.history['loss']

    plt.plot(accuracy, c='blue')
    for i in range(len(accuracy)):
        plt.scatter(i, accuracy[i], color="blue")
    plt.plot(loss, color='red')
    for i in range(len(loss)):
        plt.scatter(i, loss[i], color="red")
    plt.title('trainning history')
    plt.xlabel('epoch')
    plt.xticks(range(TRAINNING_EPOCH), range(1, TRAINNING_EPOCH + 1))
    plt.legend(['accuracy', 'loss'], loc='center right')

    plt.show()


def show_result(data, result):
    plt.figure(figsize=(9, 6))

    plt.subplot(3, 1, 1)
    plt.xticks(range(SEQUENCE_SIZE), range(1, SEQUENCE_SIZE + 1))
    plt.ylabel('seq 1')
    plt.bar(range(SEQUENCE_SIZE), data[0])

    plt.subplot(3, 1, 2)
    plt.xticks(range(SEQUENCE_SIZE), range(1, SEQUENCE_SIZE + 1))
    plt.ylabel('seq 2')
    plt.bar(range(SEQUENCE_SIZE), data[1])

    plt.subplot(3, 1, 3)
    plt.xticks(range(SEQUENCE_SIZE), range(1, SEQUENCE_SIZE + 1))
    plt.ylabel('xor')
    plt.bar(range(SEQUENCE_SIZE), result)

    plt.show()


if __name__ == '__main__':
    trainning_seq_pairs, trainning_labels = load_trainning_date()

    model = create_model()
    history = model.fit(trainning_seq_pairs,
                        trainning_labels, epochs=TRAINNING_EPOCH)
    show_history(history)

    test_seq_pairs, test_labels = load_test_date()
    test_loss, test_acc = model.evaluate(test_seq_pairs, test_labels)
    print('\nTest accuracy: {}'.format(test_acc))
    print('Test loss: {}\n'.format(test_loss))

    data = random_seq_pairs(1)
    result = model.predict(data)

    show_result(data[0], result[0])
