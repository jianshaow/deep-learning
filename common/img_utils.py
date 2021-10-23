import os
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

DATA_SET_PATH = path.join(path.expanduser('~'), '.dataset')
CIRCLE_COUNT_DATA_FILE = path.join(DATA_SET_PATH, 'circle_count')


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
    x = np.zeros((size, 100, 100), dtype=np.uint8)
    reg_y = np.zeros((size), dtype=np.uint8)
    cls_y = np.zeros((size, CIRCLES_MAX), dtype=np.uint8)

    def handle(index, images, circles):
        x[index] = images
        reg_y[index] = circles
        cls_y[index][circles] = 1
        if size >= 1000 and (index + 1) % 1000 == 0:
            print(index + 1, 'data generated...')
    random_circles_images(handle, size)
    return x, reg_y, cls_y


def save_circle_count_dataset():
    print('start to generate train data')
    train_x, train_reg_y, train_cls_y = gen_circle_count_data(
        TRAIN_DATA_SIZE)
    print('start to generate test data')
    test_x, test_reg_y, test_cls_y = gen_circle_count_data(
        TEST_DATA_SIZE)
    if not path.exists(DATA_SET_PATH):
        os.makedirs(DATA_SET_PATH)
    np.savez(CIRCLE_COUNT_DATA_FILE, train_x=train_x,
             train_reg_y=train_reg_y, train_cls_y=train_cls_y, test_x=test_x, test_reg_y=test_reg_y, test_cls_y=test_cls_y)


def load_data():
    with np.load(CIRCLE_COUNT_DATA_FILE + '.npz') as f:
        train_x = f['train_x']
        train_reg_y = f['train_reg_y']
        train_cls_y = f['train_cls_y']
        test_x = f['test_x']
        test_reg_y = f['test_reg_y']
        test_cls_y = f['test_cls_y']
        return (train_x, train_reg_y, train_cls_y), (test_x, test_reg_y, test_cls_y)


def load_cls_data():
    (train_x, _, train_y), (test_x, _, test_y) = load_data()
    return (train_x, train_y), (test_x, test_y)


def load_reg_data():
    (train_x, train_y, _), (test_x, test_y, _) = load_data()
    return (train_x, train_y), (test_x, test_y)


def show_images(images, labels, title='data', class_mapping=None, randomized=False):
    fig = plt.figure(figsize=(8, 10))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)
    if randomized:
        start = random.randint(0, len(images) - 20 - 1)
    else:
        start = 0
    fig.suptitle(title + ' [' + str(start) + ' - ' + str(start + 20) + ']')
    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.imshow(images[start + i], cmap=plt.cm.binary)
        label = labels[start + i]
        print(str(i) + ': ', label)
        xlabel = label if class_mapping == None else class_mapping[label]
        ax.set_xlabel(xlabel)
    plt.show()


def show_image(image, label, title='image', class_mapping=None):
    fig = plt.figure(figsize=(5, 6))
    fig.suptitle(title)
    ax = fig.add_axes([.05, .05, .9, .9])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.imshow(image, cmap=plt.cm.binary)
    xlabel = label if class_mapping == None else class_mapping[label]
    ax.set_xlabel(xlabel)
    plt.show()


def __show_data():
    (train_x, train_reg_y, train_cls_y), (test_x,
                                          test_reg_y, test_cls_y) = load_data()
    print(train_x.shape, train_x.dtype)
    print(train_reg_y[0], train_cls_y[0])
    print(test_x.shape, test_x.dtype)
    print(test_reg_y[0], test_cls_y[0])

    show_images(train_x, train_reg_y, title='train data', randomized=True)
    show_images(test_x, test_reg_y, title='test data', randomized=True)
    index = random.randint(0, len(train_x) - 1)
    show_image(train_x[index], train_reg_y[index])
    index = random.randint(0, len(test_x) - 1)
    show_image(test_x[index], test_reg_y[index])


if __name__ == '__main__':
    # save_circle_count_dataset()
    __show_data()
