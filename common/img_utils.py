import os
import random

import matplotlib.patches as ptchs
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import image as tfimg

TRAIN_DATA_SIZE = 5000
TEST_DATA_SIZE = 500
TRAIN_EPOCH = 10
CIRCLES_MAX = 6

SIDE_LIMIT = 100
RADIUS = 10
SPACE = 2
CENTER_LOWER = RADIUS + SPACE
CENTER_UPPER = SIDE_LIMIT - RADIUS - SPACE

CIRCLE_COUNT_DATA_FILE = os.path.join(
    os.path.expanduser('~'), '.dataset/circle_count')


def random_circles_images(handle, size=1):
    fig = plt.figure(figsize=(1, 1))
    for i in range(size):
        circles = random.randint(0, CIRCLES_MAX - 1)
        data = random_circles_image(fig, circles)
        handle(i, data, circles)
    plt.close(fig)


def random_circles_image(fig, circles):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, SIDE_LIMIT)
    ax.set_ylim(0, SIDE_LIMIT)

    centers = []
    for _i in range(circles):
        center = random_circle_center(
            centers, RADIUS, CENTER_LOWER, CENTER_UPPER)
        centers.append(center)
        circle = ptchs.Circle(center, RADIUS, fill=False)
        ax.add_artist(circle)

    canvas = fig.canvas
    canvas.draw()
    fig_size = fig.get_size_inches()
    fig_dpi = fig.get_dpi()
    # print(fig_size, fig_dpi)
    width, height = fig_size * fig_dpi
    data = np.fromstring(canvas.tostring_rgb(), np.uint8).reshape(
        (int(height), int(width), 3))
    data = tfimg.rgb_to_grayscale(data)
    data = np.squeeze(data)
    fig.clf()
    return data


def random_circle_center(centers, radius, lower, upper):
    x, y = 0, 0
    while(True):
        x = random.randint(lower, upper)
        y = random.randint(lower, upper)
        success = True
        for center in centers:
            if np.sqrt(np.square(x-center[0]) + np.square(y-center[1])) <= 2 * radius + SPACE:
                success = False
                break
        if success:
            break
    return x, y


def gen_circle_count_data(size=1):
    datas = np.zeros((size, 100, 100), dtype=np.uint8)
    reg_labels = np.zeros((size), dtype=np.uint8)
    cls_labels = np.zeros((size, CIRCLES_MAX), dtype=np.uint8)

    def handle(i, data, label):
        datas[i] = data
        reg_labels[i] = label
        cls_labels[i][label] = 1
    random_circles_images(handle, size)
    return datas, reg_labels, cls_labels


def save_circle_count_dataset():
    train_data, train_reg_label, train_cls_label = gen_circle_count_data(
        TRAIN_DATA_SIZE)
    test_data, test_reg_label, test_cls_label = gen_circle_count_data(
        TEST_DATA_SIZE)
    np.savez(CIRCLE_COUNT_DATA_FILE, train_data=train_data,
             train_reg_label=train_reg_label, train_cls_label=train_cls_label, test_data=test_data, test_reg_label=test_reg_label, test_cls_label=test_cls_label)


def load_data():
    with np.load(CIRCLE_COUNT_DATA_FILE + '.npz') as f:
        train_data = f['train_data']
        train_reg_label = f['train_reg_label']
        train_cls_label = f['train_cls_label']
        test_data = f['test_data']
        test_reg_label = f['test_reg_label']
        test_cls_label = f['test_cls_label']
        return (train_data, train_reg_label, train_cls_label), (test_data, test_reg_label, test_cls_label)


def load_cls_data():
    (train_data, _, train_label), (test_data, _, test_label) = load_data()
    return (train_data, train_label), (test_data, test_label)


def load_reg_data():
    (train_data, train_label, _), (test_data, test_label, _) = load_data()
    return (train_data, train_label), (test_data, test_label)


def show_data(x_train, y_train, class_mapping=None):
    plt.figure(figsize=(10, 10))
    start = random.randint(0, TRAIN_DATA_SIZE - 25 - 1)
    plt.suptitle('data[' + str(start) + ' - ' + str(start + 25) + ']')
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[start+i], cmap=plt.cm.binary)
        xlabel = y_train[start +
                         i] if class_mapping == None else class_mapping[y_train[start+i]]
        plt.xlabel(xlabel)
    plt.show()


def show_one(x, y, class_mapping=None):
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x, cmap=plt.cm.binary)
    xlabel = y if class_mapping == None else class_mapping[y]
    plt.xlabel(xlabel)
    plt.show()


def __test():
    (train_data, train_reg_label, train_cls_label), (test_data,
                                                     test_reg_label, test_cls_label) = load_data()
    print(train_data.shape, train_data.dtype)
    print(train_reg_label[0], train_cls_label[0])
    print(test_data.shape, test_data.dtype)
    print(test_reg_label[0], test_cls_label[0])

    show_data(train_data, train_reg_label)
    show_one(train_data[1], train_reg_label[1])


if __name__ == '__main__':
    # save_circle_count_dataset()
    __test()
