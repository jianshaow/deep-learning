import os
import shutil
from os import path

from common import img_utils as img
from common import vis_utils as vis
from common.img_utils import CIRCLES_MAX, TRAIN_EPOCH
from tensorflow import keras

MODEL_NAME = 'circle_count'
MODEL_BASE_DIR = path.join(path.expanduser('~'), '.model')

ERROR_BATCH_SIZE = 100
ERROR_DATA_SIZE = 1000

DATA_CONFIG = img.data_config(6)


def get_model_path(name):
    return path.join(MODEL_BASE_DIR, name), path.join(MODEL_BASE_DIR, name + '.old')


def prepare_data():
    (x_train, y_train), (x_test, y_test) = img.load_cls_data(DATA_CONFIG)
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, y_train, (x_test, y_test)


def prepare_error_data():
    x_train, y_train = img.load_cls_error_data(DATA_CONFIG)
    x_train = x_train/255.0
    return x_train, y_train, None


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 100)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(CIRCLES_MAX, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model


def train_model(model, x_train, y_train, epochs=TRAIN_EPOCH, validation_data=None):
    callback = vis.VisualizationCallback(
        show_model=True, show_metrics=True, dynamic_plot=True)
    model.fit(x_train, y_train, epochs=epochs,
              validation_data=validation_data,
              callbacks=[callback])


def verify_model(model, data=img.random_circles_data(DATA_CONFIG, size=20)):
    images, nums = data
    estimations = model.predict(images)
    img.show_images(images, nums, estimations, title='predict result')


def save_model(model, name=MODEL_NAME):
    model_path, model_old_path = get_model_path(name)
    if path.exists(model_path):
        if path.exists(model_old_path):
            shutil.rmtree(model_old_path)
        os.rename(model_path, model_old_path)
    model.save(model_path)


def load_model(name=MODEL_NAME):
    model_path, _ = get_model_path(name)
    return keras.models.load_model(model_path)


def first_run(model_name=MODEL_NAME, data=prepare_data(), dry_run=True):
    x_train, y_train, validation_data = data
    model = build_model()
    train_model(model, x_train, y_train, validation_data=validation_data)
    verify_model(model)

    if not dry_run:
        save_model(model, model_name)

    return model


def re_run(model_name=MODEL_NAME, model_save_as=None, data=prepare_data(), learning_rate=0.0001, epochs=100):
    x_train, y_train, validation_data = data
    model = load_model(model_name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    train_model(model, x_train, y_train, epochs=epochs,
                validation_data=validation_data)
    verify_model(model)

    save = input('save model? (y|n): ')
    if save == 'y':
        save_model_name = model_save_as if model_save_as else model_name
        save_model(model, save_model_name)

    return model


def build_error_data(model_name=MODEL_NAME, dry_run=False):
    model = load_model(model_name)

    x, reg_y, cls_y = img.zero_data(ERROR_DATA_SIZE)

    added = 0
    handled = 0
    while(added < ERROR_DATA_SIZE):
        images, circle_nums = img.random_circles_data(
            DATA_CONFIG, ERROR_BATCH_SIZE)
        estimation = model.predict(images)
        for i in range(ERROR_BATCH_SIZE):
            if estimation[i][circle_nums[i]] == 0:
                if dry_run:
                    print(estimation[i], circle_nums[i])
                x[added] = images[i]
                reg_y[added] = circle_nums[i]
                cls_y[added][circle_nums[i]] = 1
                added += 1
                if added >= ERROR_DATA_SIZE:
                    break
        handled += ERROR_BATCH_SIZE
        print(added, 'error data added per', handled)

    if not dry_run:
        img.save_error_data((x, reg_y, cls_y), DATA_CONFIG)


def demo_model(model_name=MODEL_NAME, data=img.random_circles_data(DATA_CONFIG, size=20)):
    model = load_model(model_name)
    verify_model(model, data)


def load_sample_data():
    (x_train, y_train), _ = img.load_reg_data(DATA_CONFIG)
    return x_train[:20], y_train[:20]


def load_sample_error_data():
    x_train, y_train = img.load_reg_error_data(DATA_CONFIG)
    return x_train[:20], y_train[:20]


if __name__ == '__main__':
    # first_run(model_name='circle_count.h2-64', dry_run=False)
    # re_run(data=prepare_error_data())
    re_run(model_name='circle_count.h2-64')
    # re_run(model_name='circle_count.h2-64', data=prepare_error_data())
    # demo_model(model_name='circle_count.h2-64')
    # demo_model(model_name='circle_count.h2-64', data=load_sample_error_data())
    # build_error_data(model_name='circle_count.h2-64', dry_run=True)
    # build_error_data(model_name='circle_count.h2-64')
    # build_error_data(dry_run=True)
