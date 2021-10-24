import os
import shutil
from os import path

from common import img_utils as img
from common import vis_utils as vis
from common.img_utils import CIRCLES_MAX, TRAIN_EPOCH
from tensorflow import keras

MODEL_PATH = path.join(path.expanduser('~'), '.model/circle_count')
MODEL_PATH_OLD = MODEL_PATH + '.old'

ERROR_BATCH_SIZE = 100
ERROR_DATA_SIZE = 1000

DATA_CONFIG = img.RANDOM_R_CONFIG


def prepare_data():
    (train_x, train_y), (test_x, test_y) = img.load_cls_data(DATA_CONFIG)
    train_x, test_x = train_x/255.0, test_x/255.0
    return train_x, train_y, (test_x, test_y)


def prepare_error_data():
    train_x, train_y = img.load_cls_error_data(DATA_CONFIG)
    train_x = train_x/255.0
    return train_x, train_y, None


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 100)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(CIRCLES_MAX, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model


def train_model(model, train_x, train_y, epochs=TRAIN_EPOCH, validation_data=None):
    callback = vis.VisualizationCallback(
        show_model=True, show_metrics=True, dynamic_plot=True)
    model.fit(train_x, train_y, epochs=epochs,
              validation_data=validation_data,
              callbacks=[callback])


def verify_model(model):
    images, _ = img.random_circles_data(DATA_CONFIG, size=20)
    circle_nums_estimation = model.predict(images)
    img.show_images(images, circle_nums_estimation)


def save_model(model):
    if path.exists(MODEL_PATH):
        if path.exists(MODEL_PATH_OLD):
            shutil.rmtree(MODEL_PATH_OLD)
        os.rename(MODEL_PATH, MODEL_PATH_OLD)
    model.save(MODEL_PATH)


def load_model():
    return keras.models.load_model(MODEL_PATH)


def first_run(dry_run=True):
    train_x, train_y, validation_data = prepare_data()
    model = build_model()
    train_model(model, train_x, train_y, validation_data=validation_data)
    verify_model(model)

    if not dry_run:
        save_model(model)
 
    return model


def re_run(data=prepare_data(), learning_rate=0.0001, epochs=100):
    train_x, train_y, validation_data = data
    model = load_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    train_model(model, train_x, train_y, epochs=epochs,
                validation_data=validation_data)
    verify_model(model)

    save = input('save model? (y|n): ')
    if save == 'y':
        save_model(model)


def build_error_data(dry_run=False):
    model = load_model()

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


def demo_model():
    model = load_model()
    verify_model(model)


if __name__ == '__main__':
    first_run()
    # re_run(data=prepare_error_data())
    # re_run()
    # demo_model()
    # build_error_data()
    # build_error_data(dry_run=True)
