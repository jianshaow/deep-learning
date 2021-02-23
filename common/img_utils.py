
from os import path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 100
TRAIN_EPOCH = 10


def load_circle_count_train_data(size=TRAIN_DATA_SIZE):
    datas = np.zeros((size, 400, 400, 1), dtype=np.uint8)
    labels = np.zeros((size), dtype=np.uint8)

    def handle(i, data, label):
        datas[i] = data
        labels[i] = label
    random_circles_imgs(handle, size)
    datas = datas.reshape(size, 400, 400)
    return datas, labels


def load_circle_count_test_data(size=TEST_DATA_SIZE):
    datas = np.zeros((size, 400, 400, 1), dtype=np.uint8)
    labels = np.zeros((size), dtype=np.uint8)

    def handle(i, data, label):
        datas[i] = data
        labels[i] = label
    random_circles_imgs(handle, size)
    datas.reshape(size, 400, 400)
    return datas, labels


def random_circles_imgs(handle, size=1):
    fig = plt.figure(figsize=(4, 4))
    for i in range(size):
        circles = random.randint(1, 10)

        x = []
        y = []
        d = []

        for _i in range(circles):
            dim = random.randint(100, 10000)
            d.append(dim)
            x.append(random.randint(dim//50, 800-dim//50))
            y.append(random.randint(dim//50, 800-dim//50))

        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 800)

        ax.scatter(x, y, s=d, lw=0.5, edgecolors='black', facecolor='none')

        canvas = fig.canvas
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        data = np.fromstring(canvas.tostring_rgb(), np.uint8).reshape((int(height), int(width), 3))
        data = tf.image.rgb_to_grayscale(data)

        handle(i, data, circles)
        fig.clf()
    plt.close(fig)


if __name__ == '__main__':
    train_data, train_labels = load_circle_count_train_data(1)
    # train_data = train_data/255.0

    # print(train_data.shape, train_data.dtype)
    # data = train_data[0].reshape(400, 400)
    # data = tf.image.grayscale_to_rgb(data)
    print(train_labels[0])
    plt.figure()
    plt.imshow(train_data[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
