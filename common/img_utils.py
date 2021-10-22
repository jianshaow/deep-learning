import random
from os import path

import matplotlib.patches as ptchs
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import image as tfimg

TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000
TRAIN_EPOCH = 40
CIRCLES_MAX = 6

SIDE_LIMIT = 100
RADIUS = 10
SPACE = 2
CENTER_LOWER = RADIUS + SPACE
CENTER_UPPER = SIDE_LIMIT - RADIUS - SPACE

CIRCLE_COUNT_DATA_FILE = path.join(
    path.expanduser('~'), '.dataset/circle_count')


def random_circles_data(size=1):
    images = np.zeros((size, 100, 100), dtype=np.uint8)
    circle_nums = np.zeros((size), dtype=np.uint8)

    def handle(index, image, circle_num):
        images[index] = image
        circle_nums[index] = circle_num
    random_circles_images(handle, size)
    return images, circle_nums


def random_circles_images(handle, size=1):
    fig = plt.figure(figsize=(1, 1))
    for i in range(size):
        circle_num = random.randint(0, CIRCLES_MAX - 1)
        image = random_circles_image(fig, circle_num)
        handle(i, image, circle_num)
    plt.close(fig)


def random_circles_image(fig, circle_num):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, SIDE_LIMIT)
    ax.set_ylim(0, SIDE_LIMIT)

    centers = []
    for _i in range(circle_num):
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
    data = np.zeros((size, 100, 100), dtype=np.uint8)
    reg_labels = np.zeros((size), dtype=np.uint8)
    cls_labels = np.zeros((size, CIRCLES_MAX), dtype=np.uint8)

    def handle(index, images, circles):
        data[index] = images
        reg_labels[index] = circles
        cls_labels[index][circles] = 1
    random_circles_images(handle, size)
    return data, reg_labels, cls_labels


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


def show_images(images, labels, class_mapping=None, randomized=False):
    plt.figure(figsize=(8, 8))
    if randomized:
        start = random.randint(0, len(images) - 25 - 1)
    else:
        start = 0
    plt.suptitle('data[' + str(start) + ' - ' + str(start + 25) + ']')
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[start + i], cmap=plt.cm.binary)
        label = labels[start + i]
        print(str(i) + ': ', label)
        xlabel = label if class_mapping == None else class_mapping[label]
        plt.xlabel(xlabel)
    plt.show()


def show_image(x, y, class_mapping=None):
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x, cmap=plt.cm.binary)
    xlabel = y if class_mapping == None else class_mapping[y]
    plt.xlabel(xlabel)
    plt.show()


def __show_data():
    (train_data, train_reg_label, train_cls_label), (test_data,
                                                     test_reg_label, test_cls_label) = load_data()
    print(train_data.shape, train_data.dtype)
    print(train_reg_label[0], train_cls_label[0])
    print(test_data.shape, test_data.dtype)
    print(test_reg_label[0], test_cls_label[0])

    show_images(train_data, train_reg_label, randomized=True)
    show_images(test_data, test_reg_label, randomized=True)
    show_image(train_data[0], train_reg_label[0])
    show_image(test_data[0], test_reg_label[0])


if __name__ == '__main__':
    # save_circle_count_dataset()
    __show_data()
