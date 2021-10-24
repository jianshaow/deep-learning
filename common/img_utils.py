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
RADIUS = 9
RADIUS_LOWER = 6
RADIUS_UPPER = 12
SPACE = 2

DATA_SET_PATH = path.join(path.expanduser('~'), '.dataset')
DATA_NAME_PREFIX = 'circle_count'


def data_config(r_lower=RADIUS, r_upper=None):
    config = {}
    data_name = DATA_NAME_PREFIX + '_r' + str(r_lower)
    if r_upper:
        data_name = data_name + '-' + str(r_upper)
    config['name'] = data_name
    config['path'] = path.join(DATA_SET_PATH, data_name + '.npz')
    config['error_path'] = path.join(DATA_SET_PATH, data_name + '_error.npz')

    def get_config(key='radius'):
        if key == 'radius':
            if r_lower and r_upper:
                return random.randint(r_lower, r_upper)
            else:
                return r_lower
        return config[key]
    return get_config


RANDOM_R_CONFIG = data_config(RADIUS_LOWER, RADIUS_UPPER)


def random_circles_data(get_config=RANDOM_R_CONFIG, size=1):
    images = np.zeros((size, 100, 100), dtype=np.uint8)
    circle_nums = np.zeros((size), dtype=np.uint8)

    def handle(index, image, circle_num):
        images[index] = image
        circle_nums[index] = circle_num
    random_circles_images(handle, get_config, size)
    return images, circle_nums


def random_circles_images(handle, get_config=RANDOM_R_CONFIG, size=1):
    fig = plt.figure(figsize=(1, 1))
    for i in range(size):
        circle_num = random.randint(0, CIRCLES_MAX - 1)
        image = random_circles_image(fig, circle_num, get_config)
        handle(i, image, circle_num)
    plt.close(fig)


def random_circles_image(fig, circle_num, get_config=RANDOM_R_CONFIG):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, SIDE_LIMIT)
    ax.set_ylim(0, SIDE_LIMIT)

    circle_params = []
    for _i in range(circle_num):
        radius = get_config('radius')
        center = random_center(circle_params, radius)
        circle_param = {'r': radius, 'c': center}
        circle_params.append(circle_param)
        circle = ptchs.Circle(center, radius, fill=False)
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


def random_center(circle_params, radius):
    center_lower = radius + SPACE
    center_upper = SIDE_LIMIT - radius - SPACE
    x, y = 0, 0
    while(True):
        x = random.randint(center_lower, center_upper)
        y = random.randint(center_lower, center_upper)
        success = True
        for circle_param in circle_params:
            if np.sqrt(np.square(x-circle_param['c'][0]) + np.square(y-circle_param['c'][1])) <= radius + circle_param['r'] + SPACE:
                success = False
                break
        if success:
            break
    return x, y


def zero_data(size=1):
    x = np.zeros((size, 100, 100), dtype=np.uint8)
    reg_y = np.zeros((size), dtype=np.uint8)
    cls_y = np.zeros((size, CIRCLES_MAX), dtype=np.uint8)
    return x, reg_y, cls_y


def gen_circles_data(get_config=RANDOM_R_CONFIG, size=1):
    x, reg_y, cls_y = zero_data(size)

    def handle(index, images, circles):
        x[index] = images
        reg_y[index] = circles
        cls_y[index][circles] = 1
        if size >= 1000 and (index + 1) % 1000 == 0:
            print(index + 1, 'data generated...')
    random_circles_images(handle, get_config, size)
    return x, reg_y, cls_y


def save_data(get_config=RANDOM_R_CONFIG):
    print('start to generate train data')
    train_x, train_reg_y, train_cls_y = gen_circles_data(
        get_config, TRAIN_DATA_SIZE)
    print('start to generate test data')
    test_x, test_reg_y, test_cls_y = gen_circles_data(
        get_config, TEST_DATA_SIZE)
    if not path.exists(DATA_SET_PATH):
        os.makedirs(DATA_SET_PATH)
    np.savez(get_config('path'), train_x=train_x,
             train_reg_y=train_reg_y, train_cls_y=train_cls_y,
             test_x=test_x, test_reg_y=test_reg_y, test_cls_y=test_cls_y)


def save_error_data(error_data, get_config=RANDOM_R_CONFIG):
    train_x, train_reg_y, train_cls_y = error_data
    np.savez(get_config('error_path'), train_x=train_x,
             train_reg_y=train_reg_y, train_cls_y=train_cls_y)


def load_data(path, test_data=True):
    with np.load(path) as f:
        train_x = f['train_x']
        train_reg_y = f['train_reg_y']
        train_cls_y = f['train_cls_y']
        if test_data:
            test_x = f['test_x']
            test_reg_y = f['test_reg_y']
            test_cls_y = f['test_cls_y']
            return (train_x, train_reg_y, train_cls_y), (test_x, test_reg_y, test_cls_y)
        return (train_x, train_reg_y, train_cls_y)


def load_cls_error_data(get_config=RANDOM_R_CONFIG):
    (train_x, _, train_y) = load_data(get_config('error_path'), test_data=False)
    return train_x, train_y


def load_reg_error_data(get_config=RANDOM_R_CONFIG):
    (train_x, train_y, _) = load_data(get_config('error_path'), test_data=False)
    return train_x, train_y


def load_cls_data(get_config=RANDOM_R_CONFIG):
    (train_x, _, train_y), (test_x, _, test_y) = load_data(get_config('path'))
    return (train_x, train_y), (test_x, test_y)


def load_reg_data(get_config=RANDOM_R_CONFIG):
    (train_x, train_y, _), (test_x, test_y, _) = load_data(get_config('path'))
    return (train_x, train_y), (test_x, test_y)


def show_images(images, labels, title='data', class_mapping=None, random_sample=False):
    fig = plt.figure(figsize=(8, 10))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)
    if random_sample:
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


def show_data(get_config=RANDOM_R_CONFIG):
    (train_x, train_reg_y, train_cls_y), \
        (test_x, test_reg_y, test_cls_y) = load_data(get_config('path'))
    print(train_x.shape, train_x.dtype)
    print(train_reg_y[0], train_cls_y[0])
    print(test_x.shape, test_x.dtype)
    print(test_reg_y[0], test_cls_y[0])

    show_images(train_x, train_reg_y, title='train data', random_sample=True)
    show_images(test_x, test_reg_y, title='test data', random_sample=True)
    i = random.randint(0, len(train_x) - 1)
    show_image(train_x[i], train_reg_y[i],
               title='train image [' + str(i) + ']')
    i = random.randint(0, len(test_x) - 1)
    show_image(test_x[i], test_reg_y[i], title='test image [' + str(i) + ']')


if __name__ == '__main__':
    # save_data()
    show_data(RANDOM_R_CONFIG)
