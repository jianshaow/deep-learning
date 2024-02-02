import random
import numpy as np
from matplotlib import pyplot as plt


def zero_data(size=1):
    x = np.zeros((size, 100, 100, 3), dtype=np.uint8)
    y = np.zeros((size), dtype=np.uint8)
    return x, y


def blank_image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def cls_to_num(label):
    return np.argmax(label)


def show_images(data, preds=None, title="data", random_sample=True):
    images, labels = data
    fig = plt.figure(figsize=(8, 7))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.9)

    if len(images) > 20 and random_sample:
        start = random.randint(0, len(images) - 20)
    else:
        start = 0

    fig.suptitle(title + " [" + str(start) + " - " + str(start + 20 - 1) + "]")

    errors = 0
    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.imshow(images[start + i])

        index = start + i
        label = labels[index]
        if label.shape != ():
            label = cls_to_num(label)

        error = 0
        if preds is not None:
            pred = preds[index]
            if pred.shape != (1,):
                pred = cls_to_num(pred)
            error = abs(pred - label)
            xlabel = pred
        else:
            xlabel = label
        t = ax.set_xlabel(xlabel)

        if error != 0:
            errors += 1
            print("image[", index, "]", xlabel, "!=", label, "error =", error)
            t.set_color("r")
    plt.show()

    if preds is not None:
        print("error rate is", errors / 20)


def show_image(image, label, pred=None, title="image"):
    fig = plt.figure(figsize=(5, 6))
    fig.suptitle(title)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.imshow(image)

    if label.shape != ():
        label = cls_to_num(label)
    xlabel = label

    error = 0
    if pred is not None:
        if pred.shape != (1,):
            pred = cls_to_num(pred)
        error = abs(pred - label)
        xlabel = pred

    t = ax.set_xlabel(xlabel)
    if error != 0:
        print(xlabel, "!=", label, "error =", error)
        t.set_color("r")
    plt.show()


if __name__ == "__main__":
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    show_images((x_train, y_train))
    show_image(x_train[0], y_train[0])
