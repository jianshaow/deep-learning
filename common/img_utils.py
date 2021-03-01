
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

TRAIN_DATA_SIZE = 5000
TEST_DATA_SIZE = 500
TRAIN_EPOCH = 10
CIRCLES_MAX = 5

CIRCLE_COUNT_DATA_FILE = os.path.join(os.path.expanduser('~'), '.dataset/circle_count')


def random_circles_imgs(handle, size=1):
    fig = plt.figure(figsize=(1, 1))
    for i in range(size):
        circles = random.randint(1, CIRCLES_MAX)

        x = []
        y = []
        d = []

        for _i in range(circles):
            d.append(200)
            lower = 15
            upper = 85
            x_value = random.randint(lower, upper)
            y_value = random.randint(lower, upper)
            x.append(x_value)
            y.append(y_value)
            # print('dim =', dim, ', lower =', lower, ', upper =', upper, ', x =', x_value, ', y =', y_value)

        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        ax.scatter(x, y, s=d, lw=0.5, edgecolors='black', facecolor='none')

        canvas = fig.canvas
        canvas.draw()
        fig_size = fig.get_size_inches()
        fig_dpi = fig.get_dpi()
        # print(fig_size, fig_dpi)
        width, height = fig_size * fig_dpi
        data = np.fromstring(canvas.tostring_rgb(), np.uint8).reshape(
            (int(height), int(width), 3))
        data = tf.image.rgb_to_grayscale(data)
        data = np.squeeze(data)

        handle(i, data, circles)
        fig.clf()
    plt.close(fig)


def gen_circle_count_data(size=1):
    datas = np.zeros((size, 100, 100), dtype=np.uint8)
    reg_labels = np.zeros((size), dtype=np.uint8)
    cls_labels = np.zeros((size, CIRCLES_MAX), dtype=np.uint8)

    def handle(i, data, label):
        datas[i] = data
        reg_labels[i] = label
        cls_labels[i][label - 1] = 1
    random_circles_imgs(handle, size)
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

def __test():
    (train_data, train_reg_labels, train_cls_label), (test_data, test_reg_label, test_cls_label) = load_data()
    print(train_data.shape, train_data.dtype)
    print(train_reg_labels[0], train_cls_label[0])
    plt.imshow(train_data[0])
    plt.show()
    print(test_data.shape, test_data.dtype)
    print(test_reg_label[0], test_cls_label[0])
    plt.imshow(test_data[0])
    plt.show()


if __name__ == '__main__':
    save_circle_count_dataset()
    # __test()
