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

TRAIN_EPOCHS = 50


class Model():

    def __init__(self, params):
        self.__params = params
        self.__compiled = False
        self.__sparse = False
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

    def load(self, compile=False):
        if self.model is not None:
            raise Exception('model is initialized')

        hidden_layers, units = self.__params
        model_path, _ = _get_model_path(
            _get_model_name(hidden_layers, units))
        self.model = keras.models.load_model(model_path, compile=compile)
        self.__compiled = compile

    def compile(self, learning_rate=LEARNING_RATE, sparse=True):
        if self.model is None:
            raise Exception(
                'model is not initialized, call build or load method')

        self.__sparse = sparse
        if sparse:
            loss = 'sparse_categorical_crossentropy'
            # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss = 'categorical_crossentropy'

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                           loss=loss, metrics=['accuracy'])
        self.__compiled = True

    def train(self, data, epochs=TRAIN_EPOCHS, test_data=None):
        if not self.__compiled:
            raise Exception('model is not compiled yet, call compile')

        validation_data = None
        if test_data is not None:
            x_test, y_reg_test, y_cls_test = test_data
        if self.__sparse:
            x_train, y_train, _ = data
            if test_data is not None:
                validation_data = (x_test, y_reg_test)
        else:
            x_train, _, y_train = data
            if test_data is not None:
                validation_data = (x_test, y_cls_test)

        self.model.fit(x_train, y_train, epochs=epochs,
                       validation_data=validation_data,
                       callbacks=[vis.matplotlib_callback(), vis.tensorboard_callback('circle_count')])

    def predict(self, x):
        if self.model is None:
            raise Exception(
                'model is not initialized, call build or load method')

        return self.model.predict(x)

    def verify(self, data):
        if self.model is None:
            raise Exception(
                'model is not initialized, call build or load method')

        if not self.__compiled:
            raise Exception('model is not compiled yet, call compile')

        x, y_reg, y_cls = data

        predictions = self.model.predict(x)
        img.show_images(x, y_reg, predictions, title='predict result')

        if self.__sparse:
            y = y_reg
        else:
            y = y_cls

        evaluation = self.model.evaluate(x / 255.0, y)
        print('evaluation: ', evaluation)

    def save(self, ask=False):
        if ask:
            save = input('save model [' + self.model.name + ']? (y|n): ')
            if save != 'y':
                print('model [' + self.model.name + '] not saved')
                return

        model_path, model_old_path = _get_model_path(self.model.name)
        if os.path.exists(model_path):
            if os.path.exists(model_old_path):
                shutil.rmtree(model_old_path)
            os.rename(model_path, model_old_path)
        self.model.save(model_path)
        print('model [' + self.model.name + '] saved')


def _get_model_path(name):
    return os.path.join(MODEL_BASE_DIR, name), os.path.join(MODEL_BASE_DIR, name + '.old')


def _get_model_name(hidden_layers, units):
    return MODEL_NAME_PREFIX + '.' + str(hidden_layers) + '-' + str(units)


if __name__ == '__main__':
    images, nums, _ = img.zero_data(20)

    def handle(i, image, num):
        images[i] = image
        nums[i] = num
    img.random_circles_images(handle, size=20)

    model = Model(MODEL_PARAMS)
    model.load()
    predictions = model.predict(images)
    img.show_images(images, nums, predictions)
