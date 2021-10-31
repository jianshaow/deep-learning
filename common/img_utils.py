import random

import matplotlib.patches as ptchs
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import image as tfimg

TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000
CIRCLES_MAX = 6

SIDE_LIMIT = 100
DEFAULT_RADIUS = 6
SPACE = 2


def __get_radius():
    return DEFAULT_RADIUS


def random_circles_images(handle, get_radius=__get_radius, size=1):
    fig = plt.figure(figsize=(1, 1))
    for i in range(size):
        circle_num = random.randint(0, CIRCLES_MAX - 1)
        image = __random_circles_image(fig, circle_num, get_radius)
        handle(i, image, circle_num)
    plt.close(fig)


def __random_circles_image(fig, circle_num, get_radius=__get_radius):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, SIDE_LIMIT)
    ax.set_ylim(0, SIDE_LIMIT)
    ax.axis('off')

    circle_params = []
    for _ in range(circle_num):
        radius = get_radius()
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
    return np.argmax(label)


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
            print(xlabel, predictions[start + i], '!=', label)
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


if __name__ == '__main__':
    images, nums, _ = zero_data(20)

    def handle(i, image, num):
        images[i] = image
        nums[i] = num
    random_circles_images(handle, size=20)
    show_images(images, nums)
    show_image(images[0], nums[0])
