import os.path
import random
import sys

import numpy as np

from common import data_dir
from common import img_utils as img

DATA_SET_PATH = os.path.join(data_dir, 'dataset')
DATA_NAME_PREFIX = 'circle_count'

DATA_SIZE = 100000

TOLERANCE = 0.1
ERROR_BATCH_SIZE = 100
ERROR_DATA_SIZE = 10000

CIRCLES_MAX = 6


def data_config(r_lower, r_upper=None, circles_max=CIRCLES_MAX, error_of='error'):
    config = {}
    config['r_lower'] = r_lower
    config['circles_max'] = circles_max

    data_name = '%s.%02d' % (DATA_NAME_PREFIX, r_lower)
    if r_upper is not None:
        config['r_upper'] = r_upper
        data_name = '%s-%02d' % (data_name, r_upper)
    data_name = '%s.%02d' % (data_name, circles_max)
    config['name'] = data_name
    config['path'] = os.path.join(DATA_SET_PATH, data_name + '.npz')
    config['error_path'] = os.path.join(
        DATA_SET_PATH, data_name + '.' + error_of + '.npz'
    )

    def get_config(key='radius'):
        if key == 'radius':
            if r_lower and r_upper:
                return random.randint(r_lower, r_upper)
            else:
                return r_lower
        return config[key]

    return get_config


DEFAULT_CONFIG = data_config(6, 8)


def __save_dataset(path, data):
    x, y = data

    if not os.path.exists(DATA_SET_PATH):
        os.makedirs(DATA_SET_PATH)

    np.savez(path, x=x, y=y)


def __load_dataset(path):
    with np.load(path) as dataset:
        x = dataset['x']
        y = dataset['y']

        return (x, y)


def load_error_data(get_config=DEFAULT_CONFIG):
    return __load_dataset(get_config('error_path'))


def load_data(get_config=DEFAULT_CONFIG):
    return __load_dataset(get_config('path'))


def gen_sample_data(get_config=DEFAULT_CONFIG, size=1):
    x, y = img.zero_data(size)

    def handle(index, images, circles):
        x[index] = images
        y[index] = circles
        if size >= 1000 and (index + 1) % 1000 == 0:
            print(index + 1, 'data generated...')

    img.random_circles_images(handle, get_config, size, get_config('circles_max'))

    return x, y


def load_sample_data(get_config=DEFAULT_CONFIG, size=20):
    x, y = load_data(get_config)
    return x[:size], y[:size]


def load_sample_error_data(get_config=DEFAULT_CONFIG, size=20):
    x, y = load_error_data(get_config)
    return x[:size], y[:size]


def build_data(get_config=DEFAULT_CONFIG):
    print('generating train data...')
    x, y = gen_sample_data(get_config, DATA_SIZE)

    __save_dataset(get_config('path'), (x, y))
    print('data [' + get_config('name') + '] saved')


def build_error_data(
    model, get_config=DEFAULT_CONFIG, tolerance=TOLERANCE, append=False, dry_run=False
):
    if not dry_run:
        x, y = img.zero_data(ERROR_DATA_SIZE)

    added = 0
    handled = 0
    while added < ERROR_DATA_SIZE:
        images, circle_nums = gen_sample_data(DEFAULT_CONFIG, ERROR_BATCH_SIZE)
        preds = model.predict(images)
        for i in range(ERROR_BATCH_SIZE):
            pred = preds[i]
            if pred.shape == (1,):
                pred_circle_num = pred
            else:
                pred_circle_num = img.cls_to_num(pred)
            if abs(pred_circle_num - circle_nums[i]) > tolerance:
                if dry_run:
                    print(pred_circle_num, circle_nums[i])
                else:
                    x[added] = images[i]
                    y[added] = circle_nums[i]
                added += 1
                if added >= ERROR_DATA_SIZE:
                    break
                if random.randint(0, get_config('circles_max') - 1) == 0:
                    if dry_run:
                        print(0, 0)
                    else:
                        x[added] = img.blank_image()
                        y[added] = 0
                    added += 1
                if added >= ERROR_DATA_SIZE:
                    break
        handled += ERROR_BATCH_SIZE
        print(added, 'error data added per', handled)

    if not dry_run:
        if append:
            x_exist, y_exist = load_error_data(get_config)
            x = np.concatenate((x, x_exist))
            y = np.concatenate((y, y_exist))

        __save_dataset(get_config('error_path'), (x, y))
        print('error data [' + get_config('name') + '] saved')


def show_data(get_config=DEFAULT_CONFIG):
    __show_data(load_data(get_config), 'data')


def show_error_data(get_config=DEFAULT_CONFIG):
    __show_data(load_error_data(get_config), 'data')


def __show_data(data, title='data'):
    x, y = data

    img.show_images(data, title=title)

    i = random.randint(0, len(x) - 1)
    img.show_image(x[i], y[i], title=title + ' [' + str(i) + ']')


if __name__ == '__main__':
    mod = sys.modules['__main__']
    if len(sys.argv) == 2:
        cmd = sys.argv[1]
        if hasattr(mod, cmd):
            func = getattr(mod, cmd)
            func()
            exit(0)
    # show_data()
    # import cc_model
    # model = cc_model.load_model(cc_model.RegressionModel, cc_model.MODEL_PARAMS)
    # build_error_data(model, tolerance=0.2)
    # build_error_data(model, append=True)
    # build_error_data(dry_run=True)
    show_error_data()
