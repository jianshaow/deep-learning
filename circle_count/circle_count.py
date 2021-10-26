import os
import shutil
from os import path

from common import img_utils as img
from common import vis_utils as vis
from common.img_utils import CIRCLES_MAX, TRAIN_EPOCH
from tensorflow import keras

MODEL_NAME_PREFIX = 'circle_count'
MODEL_BASE_DIR = path.join(path.expanduser('~'), '.model')

MODEL_PARAMS = (2, 64)
LEARNING_RATE = 0.0001

ERROR_BATCH_SIZE = 100
ERROR_DATA_SIZE = 1000

DATA_CONFIG = img.data_config(6)


def get_model_path(name):
    return path.join(MODEL_BASE_DIR, name), path.join(MODEL_BASE_DIR, name + '.old')


def get_model_name(hidden_layers, units):
    return MODEL_NAME_PREFIX + '.' + str(hidden_layers) + '-' + str(units)


def prepare_data():
    (x_train, y_train), (x_test, y_test) = img.load_cls_data(DATA_CONFIG)
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, y_train, (x_test, y_test)


def prepare_error_data():
    x_train, y_train = img.load_cls_error_data(DATA_CONFIG)
    x_train = x_train/255.0
    return x_train, y_train, None


def build_model(model_params=MODEL_PARAMS, learning_rate=LEARNING_RATE):
    hidden_layers, units = model_params

    model_name = get_model_name(hidden_layers, units)
    model = keras.Sequential([keras.layers.Flatten(
        input_shape=(100, 100))], name=model_name)
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(units, activation='relu'))
    model.add(keras.layers.Dense(CIRCLES_MAX, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model


def train_model(model, x_train, y_train, epochs=TRAIN_EPOCH, validation_data=None):
    callback = vis.VisualizationCallback(
        show_model=True, show_metrics=True, dynamic_plot=True)

    model.fit(x_train, y_train, epochs=epochs,
              validation_data=validation_data,
              callbacks=[callback])


def verify_model(model, data=img.gen_circles_data(DATA_CONFIG, size=20)):
    images, nums, num_labels = data

    predictions = model.predict(images)
    img.show_images(images, nums, predictions, title='predict result')

    evaluation = model.evaluate(images, num_labels)
    print('evaluation: ', evaluation)


def save_model(model):
    model_path, model_old_path = get_model_path(model.name)
    if path.exists(model_path):
        if path.exists(model_old_path):
            shutil.rmtree(model_old_path)
        os.rename(model_path, model_old_path)
    model.save(model_path)
    print('model[' + model.name + '] saved')


def load_model(model_params=MODEL_PARAMS):
    hidden_layers, units = model_params
    model_path, _ = get_model_path(get_model_name(hidden_layers, units))
    return keras.models.load_model(model_path)


def first_run(model_params=MODEL_PARAMS, data=prepare_data(), dry_run=False):
    x_train, y_train, validation_data = data
    model = build_model(model_params)
    train_model(model, x_train, y_train, validation_data=validation_data)
    verify_model(model)

    if not dry_run:
        save_model(model)

    return model


def re_run(model_params=MODEL_PARAMS, data=prepare_data(), learning_rate=None, epochs=TRAIN_EPOCH):
    x_train, y_train, validation_data = data

    model = load_model(model_params)

    if learning_rate is not None:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

    train_model(model, x_train, y_train, epochs=epochs,
                validation_data=validation_data)

    verify_model(model)

    save = input('save model[' + model.name + ']? (y|n): ')
    if save == 'y':
        save_model(model)

    return model


def build_error_data(model_params=MODEL_PARAMS, dry_run=False):
    model = load_model(model_params)

    x, y_reg, y_cls = img.zero_data(ERROR_DATA_SIZE)

    added = 0
    handled = 0
    while(added < ERROR_DATA_SIZE):
        images, circle_nums, _ = img.gen_circles_data(
            DATA_CONFIG, ERROR_BATCH_SIZE)
        estimation = model.predict(images)
        for i in range(ERROR_BATCH_SIZE):
            if estimation[i][circle_nums[i]] == 0:
                if dry_run:
                    print(estimation[i], circle_nums[i])
                x[added] = images[i]
                y_reg[added] = circle_nums[i]
                y_cls[added][circle_nums[i]] = 1
                added += 1
                if added >= ERROR_DATA_SIZE:
                    break
        handled += ERROR_BATCH_SIZE
        print(added, 'error data added per', handled)

    if not dry_run:
        img.save_error_data((x, y_reg, y_cls), DATA_CONFIG)


def demo_model(model_params=MODEL_PARAMS, data=img.gen_circles_data(DATA_CONFIG, size=20)):
    model = load_model(model_params)
    verify_model(model, data)


def load_sample_data(size=20):
    (x_train, y_reg_train, y_cls_train), _ = img.load_cls_data(DATA_CONFIG)
    return x_train[:size], y_reg_train[:size], y_cls_train[:size]


def load_sample_error_data(size=20):
    x_train, y_reg_train, y_cls_train = img.load_error_data(DATA_CONFIG)
    return x_train[:size], y_reg_train[:size], y_cls_train[:size]


if __name__ == '__main__':
    # first_run()
    # first_run(dry_run=False)
    # re_run()
    # re_run(data=prepare_error_data())
    # demo_model()
    demo_model(data=load_sample_error_data(1000))
    # build_error_data()
    # build_error_data(dry_run=True)
