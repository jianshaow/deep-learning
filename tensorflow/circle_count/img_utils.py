import random

import matplotlib.patches as ptchs
import matplotlib.pyplot as plt
import numpy as np

from circle_count import DEFAULT_CIRCLES_MAX

SIDE_LIMIT = 100
DEFAULT_RADIUS = 6
SPACE = 2
TOLERANCE = 0.1


def __get_radius():
    return DEFAULT_RADIUS


def random_circles_images(
    handle, get_radius=__get_radius, size=1, circles_max=DEFAULT_CIRCLES_MAX
):
    fig = plt.figure(figsize=(1, 1))
    for i in range(size):
        circle_num = random.randint(0, circles_max - 1)
        image = __random_circles_image(fig, circle_num, get_radius)
        handle(i, image, circle_num)
    plt.close(fig)


def __random_circles_image(fig, circle_num, get_radius=__get_radius):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, SIDE_LIMIT)
    ax.set_ylim(0, SIDE_LIMIT)
    ax.axis("off")

    circle_params = []
    for _ in range(circle_num):
        radius = get_radius()
        center = __random_center(circle_params, radius)
        circle_param = {"r": radius, "c": center}
        circle_params.append(circle_param)
        circle = ptchs.Circle(center, radius, fill=False)
        ax.add_artist(circle)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb_array = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    data = rgb_array.reshape((height, width, 4))[:, :, :3]

    fig.clf()
    return data


def __random_center(circle_params, radius):
    center_lower = radius + SPACE
    center_upper = SIDE_LIMIT - radius - SPACE
    x, y = 0, 0
    while True:
        x = random.randint(center_lower, center_upper)
        y = random.randint(center_lower, center_upper)
        accepted = True
        for circle_param in circle_params:
            center_x, center_y = circle_param["c"]
            distance = np.sqrt(np.square(x - center_x) + np.square(y - center_y))
            if distance <= radius + circle_param["r"] + SPACE:
                accepted = False
                break
        if accepted:
            break
    return x, y


def zero_data(size=1):
    x = np.zeros((size, 100, 100, 3), dtype=np.uint8)
    y = np.zeros((size), dtype=np.uint8)
    return x, y


def blank_image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def cls_to_num(label):
    return np.argmax(label)


def show_images(
    data, preds=None, title="data", tolerance=TOLERANCE, random_sample=True
):
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
        ax.imshow(images[start + i], cmap=plt.cm.binary, vmin=0, vmax=255)

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

        if error > tolerance:
            errors += 1
            print("image[", index, "]", xlabel, "!=", label, "error =", error)
            t.set_color("r")
    plt.show()

    if preds is not None:
        print("error rate is", errors / 20, "with tolerance", tolerance)


def show_image(image, label, pred=None, title="image", tolerance=TOLERANCE):
    fig = plt.figure(figsize=(5, 6))
    fig.suptitle(title)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.imshow(image, cmap=plt.cm.binary, vmin=0, vmax=255)

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
    if error > tolerance:
        print(xlabel, "!=", label, "error =", error)
        t.set_color("r")
    plt.show()


if __name__ == "__main__":
    images, nums = zero_data(20)

    def handle(i, image, num):
        images[i] = image
        nums[i] = num

    random_circles_images(handle, size=20)
    show_images((images, nums))
    show_image(images[0], nums[0])
