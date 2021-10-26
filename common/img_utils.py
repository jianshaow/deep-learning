import os
import random
from os import path

import matplotlib.patches as ptchs
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import image as tfimg

TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000
TRAIN_EPOCH = 100
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
    config['error_path'] = path.join(DATA_SET_PATH, data_name + '.error.npz')

    def get_config(key='radius'):
        if key == 'radius':
            if r_lower and r_upper:
                return random.randint(r_lower, r_upper)
            else:
                return r_lower
        return config[key]
    return get_config


RANDOM_R_CONFIG = data_config(RADIUS_LOWER, RADIUS_UPPER)


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
    ax.axis('off')

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


def cls_to_num(labels):
    num = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            num = i
    return num


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
    x_train, y_reg_train, y_cls_train = gen_circles_data(
        get_config, TRAIN_DATA_SIZE)
    print('start to generate test data')
    x_test, y_reg_test, y_cls_test = gen_circles_data(
        get_config, TEST_DATA_SIZE)
    if not path.exists(DATA_SET_PATH):
        os.makedirs(DATA_SET_PATH)
    np.savez(get_config('path'), x_train=x_train,
             y_reg_train=y_reg_train, y_cls_train=y_cls_train,
             x_test=x_test, y_reg_test=y_reg_test, y_cls_test=y_cls_test)


def save_error_data(error_data, get_config=RANDOM_R_CONFIG):
    x_train, y_reg_train, y_cls_train = error_data
    np.savez(get_config('error_path'), x_train=x_train,
             y_reg_train=y_reg_train, y_cls_train=y_cls_train)


def load_data(path, test_data=True):
    with np.load(path) as data:
        x_train = data['x_train']
        y_reg_train = data['y_reg_train']
        y_cls_train = data['y_cls_train']
        if test_data:
            x_test = data['x_test']
            y_reg_test = data['y_reg_test']
            y_cls_test = data['y_cls_test']
            return (x_train, y_reg_train, y_cls_train), (x_test, y_reg_test, y_cls_test)
        return (x_train, y_reg_train, y_cls_train)


def load_error_data(get_config=RANDOM_R_CONFIG):
    return load_data(get_config('error_path'), test_data=False)


def load_cls_error_data(get_config=RANDOM_R_CONFIG):
    (x_train, _, y_train) = load_error_data(get_config)
    return x_train, y_train


def load_reg_error_data(get_config=RANDOM_R_CONFIG):
    (x_train, y_train, _) = load_error_data(get_config)
    return x_train, y_train


def load_normal_data(get_config=RANDOM_R_CONFIG):
    return load_data(get_config('path'))


def load_cls_data(get_config=RANDOM_R_CONFIG):
    (x_train, _, y_train), (x_test, _, y_test) = load_normal_data(get_config)
    return (x_train, y_train), (x_test, y_test)


def load_reg_data(get_config=RANDOM_R_CONFIG):
    (x_train, y_train, _), (x_test, y_test, _) = load_normal_data(get_config)
    return (x_train, y_train), (x_test, y_test)


def show_images(images, labels, predictions=None, title='data'):
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)

    if len(images) > 20:
        start = random.randint(0, len(images) - 20)
    else:
        start = 0

    fig.suptitle(title + ' [' + str(start) + ' - ' + str(start + 20 - 1) + ']')

    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.imshow(images[start + i], cmap=plt.cm.binary, vmin=0, vmax=255)

        label = labels[start + i]
        if label.shape == (CIRCLES_MAX,):
            label = cls_to_num(label)
        xlabel = label

        prediction = None
        if predictions is not None:
            prediction = predictions[start + i]
            if prediction.shape == (CIRCLES_MAX,):
                prediction = cls_to_num(prediction)
            xlabel = prediction

        t = ax.set_xlabel(xlabel)
        if prediction is not None and prediction != label:
            print(str(i) + ': ', predictions[start + i])
            t.set_color('r')
    plt.show()


def show_image(image, label, estimation=None, title='image'):
    fig = plt.figure(figsize=(5, 6))
    fig.suptitle(title)
    ax = fig.add_axes([.05, .05, .9, .9])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.imshow(image, cmap=plt.cm.binary, vmin=0, vmax=255)

    if label.shape == (CIRCLES_MAX,):
        label = cls_to_num(label)
    xlabel = label

    if estimation is not None:
        if estimation.shape == (CIRCLES_MAX,):
            estimation = cls_to_num(estimation)
        xlabel = estimation

    t = ax.set_xlabel(xlabel)
    if estimation is not None and estimation != label:
        t.set_color('r')
    plt.show()


def show_data(get_config=RANDOM_R_CONFIG):
    (x_train, y_reg_train, y_cls_train), \
        (x_test, y_reg_test, y_cls_test) = load_data(get_config('path'))
    print(x_train.shape, x_train.dtype)
    print(y_reg_train[0], y_cls_train[0])
    print(x_test.shape, x_test.dtype)
    print(y_reg_test[0], y_cls_test[0])

    show_images(x_train, y_reg_train, title='train data')
    show_images(x_test, y_reg_test, title='test data')
    i = random.randint(0, len(x_train) - 1)
    show_image(x_train[i], y_reg_train[i],
               title='train image [' + str(i) + ']')
    i = random.randint(0, len(x_test) - 1)
    show_image(x_test[i], y_reg_test[i], title='test image [' + str(i) + ']')


if __name__ == '__main__':
    # save_data()
    show_data(data_config(6))
