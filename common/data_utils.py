import random

import numpy as np

SEQUENCE_SIZE = 10
TRAIN_DATA_SIZE = 5000
TEST_DATA_SIZE = 500
TRAIN_EPOCH = 10


def __seq_xor(seq_pair):
    xors = np.zeros((seq_pair.shape[1]), dtype=np.uint8)

    for i in range(len(seq_pair[0])):
        xors[i] = (seq_pair[0][i] ^ seq_pair[1][i])
    return xors


def __random_seq():
    seq = np.zeros((SEQUENCE_SIZE), dtype=np.uint8)
    for i in range(SEQUENCE_SIZE):
        seq[i] = random.randint(0, 1)
    return seq


def batch_xor(seqs):
    result = np.zeros((seqs.shape[0], SEQUENCE_SIZE), dtype=np.uint8)
    for i in range(len(seqs)):
        result[i] = __seq_xor(seqs[i])
    return result


def random_seq_pairs(size=10):
    seq_pairs = np.zeros((size, 2, SEQUENCE_SIZE), dtype=np.uint8)
    for i in range(size):
        seq_pairs[i][0] = __random_seq()
        seq_pairs[i][1] = __random_seq()
    return seq_pairs


def gen_xor_train_data(size=TRAIN_DATA_SIZE):
    train_seq_pairs = random_seq_pairs(size)
    train_labels = batch_xor(train_seq_pairs)
    return train_seq_pairs, train_labels


def gen_xor_test_data(size=TEST_DATA_SIZE):
    test_seq_pairs = random_seq_pairs(size)
    test_labels = batch_xor(test_seq_pairs)
    return test_seq_pairs, test_labels
