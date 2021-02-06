
from os import path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tempfile
import uuid

TRAIN_DATA_SIZE = 100
TEST_DATA_SIZE = 10
TRAIN_EPOCH = 10


def gen_circle_count_train_data(size=TRAIN_DATA_SIZE):
    x = np.zeros((size, 400, 400, 3), dtype=np.uint8)
    y = np.zeros((size), dtype=np.uint8)
    imgs = random_circles_imgs(size)
    for i in range(len(imgs)):
        x[i], y[i] = imgs[i]
    return x, y


def gen_circle_count_test_data(size=TEST_DATA_SIZE):
    x = np.zeros((size, 400, 400, 3), dtype=np.uint8)
    y = np.zeros((size), dtype=np.uint8)
    imgs = random_circles_imgs(size)
    for i in range(len(imgs)):
        x[i], y[i] = imgs[i]
    return x, y


def random_circles_imgs(size=1):
    result = []
    fig = plt.figure(figsize=(4, 4))
    with tempfile.TemporaryDirectory() as tmpdirname:
        for _i in range(size):
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
            image_file = path.join(tmpdirname, str(
                uuid.uuid1()) + '-' + str(circles).rjust(2, '0') + '.png')
            fig.savefig(image_file)
            raw_file = tf.io.read_file(image_file)

            data = tf.image.decode_image(raw_file, channels=3)
            result.append((data, circles))
            fig.clf()
    plt.close(fig)

    return result


if __name__ == '__main__':
    train_data, train_labels = gen_circle_count_train_data(1)
    train_data = train_data/255.0

    print(train_data.shape, train_data.dtype)



