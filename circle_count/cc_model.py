import os
import shutil

from common import img_utils as img
from common import vis_utils as vis
from common.img_utils import CIRCLES_MAX
from tensorflow import keras

MODEL_NAME_PREFIX = 'circle_count'
MODEL_BASE_DIR = os.path.join(os.path.expanduser('~'), '.model')

MODEL_PARAMS = (2, 64)
LEARNING_RATE = 0.00001

TRAIN_EPOCH = 50


class Model():

    def __init__(self, params):
        self.__params = params
        self.__compiled = False
        self.model = None

    def build(self):
        if self.model is not None:
            raise Exception('model is initialized')

        hidden_layers, units = self.__params

        model_name = _get_model_name(hidden_layers, units)
        self.model = keras.Sequential(
            [keras.layers.Flatten(input_shape=(100, 100))], name=model_name)
        for _ in range(hidden_layers):
            self.model.add(keras.layers.Dense(units, activation='relu'))
        self.model.add(keras.layers.Dense(CIRCLES_MAX, activation='softmax'))

    def load(self):
        if self.model is not None:
            raise Exception('model is initialized')

        hidden_layers, units = self.__params
        model_path, _ = _get_model_path(
            _get_model_name(hidden_layers, units))
        self.model = keras.models.load_model(model_path)
        self.__compiled = True

    def compile(self, learning_rate=LEARNING_RATE):
        if self.model is None:
            raise Exception(
                'model is not initialized, call build or load method')

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.__compiled = True

    def train(self, x_train, y_train, epochs=TRAIN_EPOCH, validation_data=None):
        if not self.__compiled:
            raise Exception('model is not compiled yet, call compile')

        callback = vis.VisualizationCallback(
            show_model=True, show_metrics=True, dynamic_plot=True)

        self.model.fit(x_train, y_train, epochs=epochs,
                       validation_data=validation_data,
                       callbacks=[callback])

    def predict(self, x):
        if self.model is None:
            raise Exception(
                'model is not initialized, call build or load method')

        return self.model.predict(x)

    def verify(self, data):
        if self.model is None:
            raise Exception(
                'model is not initialized, call build or load method')

        x, y_reg, y_cls = data

        predictions = self.model.predict(x)
        img.show_images(x, y_reg, predictions, title='predict result')

        evaluation = self.model.evaluate(x, y_cls)
        print('evaluation: ', evaluation)

    def save(self, ask=False):
        if ask:
            save = input('save model[' + self.model.name + ']? (y|n): ')
            if save != 'y':
                print('model[' + self.model.name + '] not saved')
                return

        model_path, model_old_path = _get_model_path(self.model.name)
        if os.path.exists(model_path):
            if os.path.exists(model_old_path):
                shutil.rmtree(model_old_path)
            os.rename(model_path, model_old_path)
        self.model.save(model_path)
        print('model[' + self.model.name + '] saved')


def _get_model_path(name):
    return os.path.join(MODEL_BASE_DIR, name), os.path.join(MODEL_BASE_DIR, name + '.old')


def _get_model_name(hidden_layers, units):
    return MODEL_NAME_PREFIX + '.' + str(hidden_layers) + '-' + str(units)
