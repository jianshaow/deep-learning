import os
import shutil
from os import path

from common import img_utils as img
from common import vis_utils as vis
from common.img_utils import CIRCLES_MAX, TRAIN_EPOCH
from tensorflow import keras

MODEL_PATH = path.join(path.expanduser('~'), '.model/circle_count')
MODEL_PATH_OLD = MODEL_PATH + '.old'


def prepare_data():
    (train_x, train_y), (test_x, test_y) = img.load_cls_data()
    train_x, test_x = train_x/255.0, test_x/255.0
    return train_x, train_y, (test_x, test_y)


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 100)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
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
    images, _ = img.random_circles_data(20)
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


def first_run(save=False):
    train_x, train_y, validation_data = prepare_data()
    model = build_model()
    train_model(model, train_x, train_y, validation_data=validation_data)
    verify_model(model)

    if save:
        save_model(model)


def re_run():
    train_x, train_y, validation_data = prepare_data()
    model = load_model()
    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    train_model(model, train_x, train_y, validation_data=validation_data)
    verify_model(model)

    save = input('save model? (y|n): ')
    if save == 'y':
        save_model(model)


def demo_model():
    model = load_model()
    verify_model(model)


if __name__ == '__main__':
    demo_model()
