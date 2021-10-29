import os.path
import random
import sys

import numpy as np
from common import img_utils as img

import cc_model

DATA_SET_PATH = os.path.join(os.path.expanduser('~'), '.dataset')
DATA_NAME_PREFIX = 'circle_count'

TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 1000

ERROR_BATCH_SIZE = 100
ERROR_DATA_SIZE = 1000

MODEL_PARAMS = (2, 64)


def data_config(r_lower, r_upper=None):
    config = {}
    config['r_lower'] = r_lower
    data_name = DATA_NAME_PREFIX + '.' + '%02d' % r_lower
    if r_upper is not None:
        data_name = data_name + '-' + '%02d' % r_upper
        config['r_upper'] = r_upper
    config['name'] = data_name
    config['path'] = os.path.join(DATA_SET_PATH, data_name + '.npz')
    config['error_path'] = os.path.join(
        DATA_SET_PATH, data_name + '.error.npz')

    def get_config(key='radius'):
        if key == 'radius':
            if r_lower and r_upper:
                return random.randint(r_lower, r_upper)
            else:
                return r_lower
        return config[key]
    return get_config


DEFAULT_CONFIG = data_config(6, 7)


def gen_dataset(get_config=DEFAULT_CONFIG):
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


def save_error_dataset(error_data, get_config=DEFAULT_CONFIG, append=False):
    x_train, y_reg_train, y_cls_train = error_data

    if append:
        x, y_reg, y_cls = load_error_data(get_config)
        x_train = np.concatenate((x_train, x))
        y_reg_train = np.concatenate((y_reg_train, y_reg))
        y_cls_train = np.concatenate((y_cls_train, y_cls))

    __save_dataset(get_config('error_path'),
                   (x_train, y_reg_train, y_cls_train))


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


def load_error_data(get_config=DEFAULT_CONFIG):
    return __load_dataset(get_config('error_path'))


def load_cls_error_data(get_config=DEFAULT_CONFIG):
    (x_train, _, y_train) = load_error_data(get_config)
    return x_train, y_train


def load_reg_error_data(get_config=DEFAULT_CONFIG):
    (x_train, y_train, _) = load_error_data(get_config)
    return x_train, y_train


def load_data(get_config=DEFAULT_CONFIG):
    return __load_dataset(get_config('path'), test_data=True)


def load_cls_data(get_config=DEFAULT_CONFIG):
    (x_train, _, y_train), (x_test, _, y_test) = load_data(get_config)
    return (x_train, y_train), (x_test, y_test)


def load_reg_data(get_config=DEFAULT_CONFIG):
    (x_train, y_train, _), (x_test, y_test, _) = load_data(get_config)
    return (x_train, y_train), (x_test, y_test)


def gen_circles_data(get_config=DEFAULT_CONFIG, size=1):
    x, reg_y, cls_y = img.zero_data(size)

    def handle(index, images, circles):
        x[index] = images
        reg_y[index] = circles
        cls_y[index][circles] = 1
        if size >= 1000 and (index + 1) % 1000 == 0:
            print(index + 1, 'data generated...')
    img.random_circles_images(handle, get_config, size)

    return x, reg_y, cls_y


def prepare_data(get_config=DEFAULT_CONFIG):
    (x_train, y_train), (x_test, y_test) = load_cls_data(get_config)
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, y_train, (x_test, y_test)


def prepare_error_data(get_config=DEFAULT_CONFIG):
    x_train, y_train = load_cls_error_data(get_config)
    _, (x_test, y_test) = load_cls_data(get_config)
    x_train = x_train/255.0
    return x_train, y_train, (x_test[:100], y_test[:100])


def load_sample_data(get_config=DEFAULT_CONFIG, size=20):
    (x_train, y_reg_train, y_cls_train), _ = load_data(get_config)
    return x_train[:size], y_reg_train[:size], y_cls_train[:size]


def load_sample_error_data(get_config=DEFAULT_CONFIG, size=20):
    x_train, y_reg_train, y_cls_train = load_error_data(get_config)
    return x_train[:size], y_reg_train[:size], y_cls_train[:size]


def build_error_data(model_params=MODEL_PARAMS, append=False, dry_run=False):
    model = cc_model.Model(model_params)
    model.load()

    if not dry_run:
        x, y_reg, y_cls = img.zero_data(ERROR_DATA_SIZE)

    added = 0
    handled = 0
    while(added < ERROR_DATA_SIZE):
        images, circle_nums, _ = gen_circles_data(
            DEFAULT_CONFIG, ERROR_BATCH_SIZE)
        predictions = model.predict(images)
        for i in range(ERROR_BATCH_SIZE):
            if predictions[i][circle_nums[i]] == 0:
                if dry_run:
                    print(predictions[i], circle_nums[i])
                else:
                    x[added] = images[i]
                    y_reg[added] = circle_nums[i]
                    y_cls[added][circle_nums[i]] = 1
                added += 1
                if (added + 5) % 10 == 0:
                    if dry_run:
                        print(img.num_to_cls(0), 0)
                    else:
                        x[added] = img.blank_image()
                        y_reg[added] = 0
                        y_cls[added][0] = 1
                    added += 1
                if added >= ERROR_DATA_SIZE:
                    break
        handled += ERROR_BATCH_SIZE
        print(added, 'error data added per', handled)

    if not dry_run:
        save_error_dataset((x, y_reg, y_cls), DEFAULT_CONFIG, append)


def show_data(get_config=DEFAULT_CONFIG):
    (x_train, y_reg_train, y_cls_train), \
        (x_test, y_reg_test, y_cls_test) = __load_dataset(
            get_config('path'), test_data=True)

    img.show_images(x_train, y_reg_train, title='train data')
    img.show_images(x_test, y_reg_test, title='test data')

    i = random.randint(0, len(x_train) - 1)
    print(y_cls_train[i])
    img.show_image(x_train[i], y_reg_train[i],
                   title='train image [' + str(i) + ']')

    i = random.randint(0, len(x_test) - 1)
    print(y_cls_test[i])
    img.show_image(x_test[i], y_reg_test[i],
                   title='test image [' + str(i) + ']')


if __name__ == '__main__':
    mod = sys.modules['__main__']
    if len(sys.argv) == 2:
        cmd = sys.argv[1]
        if hasattr(mod, cmd):
            func = getattr(mod, cmd)
            func()
            exit(0)
    build_error_data()
    # build_error_data(append=True)
    # build_error_data(dry_run=True)
