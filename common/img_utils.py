import os
import random

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

DATA_SET_PATH = os.path.join(os.path.expanduser('~'), '.dataset')
DATA_NAME_PREFIX = 'circle_count'


def data_config(r_lower=RADIUS, r_upper=None):
    config = {}
    data_name = DATA_NAME_PREFIX + '_r' + str(r_lower)
    if r_upper:
        data_name = data_name + '-' + str(r_upper)
    config['name'] = data_name
    config['path'] = os.path.join(DATA_SET_PATH, data_name + '.npz')
    config['error_path'] = os.path.join(DATA_SET_PATH, data_name + '.error.npz')

    def get_config(key='radius'):
        if key == 'radius':
            if r_lower and r_upper:
                return random.randint(r_lower, r_upper)
            else:
                return r_lower
        return config[key]
    return get_config


RANDOM_R_CONFIG = data_config(RADIUS_LOWER, RADIUS_UPPER)


def __random_circles_images(handle, get_config=RANDOM_R_CONFIG, size=1):
    fig = plt.figure(figsize=(1, 1))
    for i in range(size):
        circle_num = random.randint(0, CIRCLES_MAX - 1)
        image = __random_circles_image(fig, circle_num, get_config)
        handle(i, image, circle_num)
    plt.close(fig)


def __random_circles_image(fig, circle_num, get_config=RANDOM_R_CONFIG):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, SIDE_LIMIT)
    ax.set_ylim(0, SIDE_LIMIT)
    ax.axis('off')

    circle_params = []
    for _ in range(circle_num):
        radius = get_config('radius')
        center = __random_center(circle_params, radius)
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


def __random_center(circle_params, radius):
    center_lower = radius + SPACE
    center_upper = SIDE_LIMIT - radius - SPACE
    x, y = 0, 0
    while(True):
        x = random.randint(center_lower, center_upper)
        y = random.randint(center_lower, center_upper)
        accepted = True
        for circle_param in circle_params:
            center_x, center_y = circle_param['c']
            distance = np.sqrt(np.square(x-center_x) + np.square(y-center_y))
            if distance <= radius + circle_param['r'] + SPACE:
                accepted = False
                break
        if accepted:
            break
    return x, y


def zero_data(size=1):
    x = np.zeros((size, 100, 100), dtype=np.uint8)
    reg_y = np.zeros((size), dtype=np.uint8)
    cls_y = np.zeros((size, CIRCLES_MAX), dtype=np.uint8)
    return x, reg_y, cls_y


def blank_image():
    return np.full((100, 100), 255, dtype=np.uint8)


def num_to_cls(num):
    label = np.zeros((CIRCLES_MAX), dtype=np.uint8)
    label[num] = 1
    return label


def cls_to_num(label):
    num = 0
    for i in range(len(label)):
        if label[i] == 1:
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
    __random_circles_images(handle, get_config, size)

    return x, reg_y, cls_y


def gen_dataset(get_config=RANDOM_R_CONFIG):
    print('generating train data...')
    x_train, y_reg_train, y_cls_train = gen_circles_data(
        get_config, TRAIN_DATA_SIZE)

    print('generating test data...')
    x_test, y_reg_test, y_cls_test = gen_circles_data(
        get_config, TEST_DATA_SIZE)

    __save_dataset(get_config('path'), (x_train, y_reg_train, y_cls_train),
                   (x_test, y_reg_test, y_cls_test))


def __save_dataset(path, train_data, test_data=None):
    x_train, y_reg_train, y_cls_train = train_data

    if not os.path.exists(DATA_SET_PATH):
        os.makedirs(DATA_SET_PATH)

    if test_data is not None:
        x_test, y_reg_test, y_cls_test = test_data
        np.savez(path, x_train=x_train, y_reg_train=y_reg_train, y_cls_train=y_cls_train,
                 x_test=x_test, y_reg_test=y_reg_test, y_cls_test=y_cls_test)
    else:
        np.savez(path, x_train=x_train, y_reg_train=y_reg_train,
                 y_cls_train=y_cls_train)


def save_error_dataset(error_data, get_config=RANDOM_R_CONFIG, append=False):
    x_train, y_reg_train, y_cls_train = error_data

    if append:
        x, y_reg, y_cls = load_error_data(get_config)
        x_train = np.concatenate((x_train, x))
        y_reg_train = np.concatenate((y_reg_train, y_reg))
        y_cls_train = np.concatenate((y_cls_train, y_cls))

    __save_dataset(get_config('error_path'), (x_train, y_reg_train, y_cls_train))


def __load_dataset(path, test_data=False):
    with np.load(path) as dataset:
        x_train = dataset['x_train']
        y_reg_train = dataset['y_reg_train']
        y_cls_train = dataset['y_cls_train']

        if test_data:
            x_test = dataset['x_test']
            y_reg_test = dataset['y_reg_test']
            y_cls_test = dataset['y_cls_test']
            return (x_train, y_reg_train, y_cls_train), (x_test, y_reg_test, y_cls_test)

        return (x_train, y_reg_train, y_cls_train)


def load_error_data(get_config=RANDOM_R_CONFIG):
    return __load_dataset(get_config('error_path'))


def load_cls_error_data(get_config=RANDOM_R_CONFIG):
    (x_train, _, y_train) = load_error_data(get_config)
    return x_train, y_train


def load_reg_error_data(get_config=RANDOM_R_CONFIG):
    (x_train, y_train, _) = load_error_data(get_config)
    return x_train, y_train


def load_data(get_config=RANDOM_R_CONFIG):
    return __load_dataset(get_config('path'), test_data=True)


def load_cls_data(get_config=RANDOM_R_CONFIG):
    (x_train, _, y_train), (x_test, _, y_test) = load_data(get_config)
    return (x_train, y_train), (x_test, y_test)


def load_reg_data(get_config=RANDOM_R_CONFIG):
    (x_train, y_train, _), (x_test, y_test, _) = load_data(get_config)
    return (x_train, y_train), (x_test, y_test)


def show_images(images, labels, predictions=None, title='data'):
    fig = plt.figure(figsize=(8, 7))
    fig.subplots_adjust(.05, .05, .95, .9)

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

        if predictions is not None:
            prediction = predictions[start + i]
            if prediction.shape == (CIRCLES_MAX,):
                prediction = cls_to_num(prediction)
            xlabel = prediction
        else:
            xlabel = label

        t = ax.set_xlabel(xlabel)
        if xlabel != label:
            print(str(i) + ': ', predictions[start + i], '!=', label)
            t.set_color('r')
    plt.show()


def show_image(image, label, prediction=None, title='image'):
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

    if prediction is not None:
        if prediction.shape == (CIRCLES_MAX,):
            prediction = cls_to_num(prediction)
        xlabel = prediction

    t = ax.set_xlabel(xlabel)
    if xlabel != label:
        t.set_color('r')
    plt.show()


def show_data(get_config=RANDOM_R_CONFIG):
    (x_train, y_reg_train, y_cls_train), \
        (x_test, y_reg_test, y_cls_test) = __load_dataset(
            get_config('path'), test_data=True)

    show_images(x_train, y_reg_train, title='train data')
    show_images(x_test, y_reg_test, title='test data')

    i = random.randint(0, len(x_train) - 1)
    print(y_cls_train[i])
    show_image(x_train[i], y_reg_train[i],
               title='train image [' + str(i) + ']')

    i = random.randint(0, len(x_test) - 1)
    print(y_cls_test[i])
    show_image(x_test[i], y_reg_test[i], title='test image [' + str(i) + ']')


if __name__ == '__main__':
    # save_dataset()
    show_data(data_config(6))
