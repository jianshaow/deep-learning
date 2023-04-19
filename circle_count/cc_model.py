import os
import shutil

from common import img_utils as img
from common import vis_utils as vis
from common.img_utils import CIRCLES_MAX
from tensorflow import keras

MODEL_NAME_PREFIX = 'circle_count'
MODEL_BASE_DIR = os.path.join(os.path.expanduser('~'), '.model')

MODEL_PARAMS = {'input_shape': (100, 100), 'hidden_layers': 2, 'hidden_layer_units': 64}
LEARNING_RATE = 0.00001

TRAIN_EPOCHS = 50


class Model:
    def __init__(self, params):
        self.__compiled = False
        self.__sparse = False
        self._params = params
        self.model = None

    def build(self):
        if self.model is not None:
            raise Exception('model is initialized')

        self._construct_model()
        self._construct_input_layer()
        self._construct_hidden_layer()
        self._construct_output_layer()

    def load(self, compile=False):
        if self.model is not None:
            raise Exception('model is initialized')

        model_path, _ = self.__get_model_path()
        self.model = keras.models.load_model(model_path, compile=compile)
        self.__compiled = compile

    def compile(self, learning_rate=LEARNING_RATE, sparse=False):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method')

        self.__sparse = sparse
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=self._get_loss(),
            metrics=self._get_metrics(),
        )
        self.__compiled = True

    def train(self, data, epochs=TRAIN_EPOCHS, test_data=None):
        if not self.__compiled:
            raise Exception('model is not compiled yet, call compile first')

        validation_data, x_train, y_train = self._extract_date(data, test_data)

        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[
                vis.matplotlib_callback(),
                vis.tensorboard_callback('circle_count'),
            ],
        )

    def predict(self, x):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method')

        return self.model.predict(x)

    def verify(self, data):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method')

        if not self.__compiled:
            raise Exception('model is not compiled yet, call compile')

        x, y_reg, y = self._extract_verification_data(data)

        predictions = self.model.predict(x)
        img.show_images(x, y_reg, predictions, title='predict result')

        evaluation = self.model.evaluate(x / 255.0, y)
        print('evaluation: ', evaluation)

    def save(self, ask=False):
        if ask:
            save = input('save model ["{}"]? (y|n): '.format(self.model.name))
            if save != 'y':
                print('model [{}] not saved'.format(self.model.name))
                return

        model_path, model_old_path = self.__get_model_path()
        if os.path.exists(model_path):
            if os.path.exists(model_old_path):
                shutil.rmtree(model_old_path)
            os.rename(model_path, model_old_path)
        self.model.save(model_path)
        print('model [' + self.model.name + '] saved')

    def _construct_model(self):
        self.model = keras.Sequential([], name=self._get_model_name())

    def _construct_input_layer(self):
        input_shape = self._params['input_shape']
        self.model.add(keras.layers.Flatten(input_shape=input_shape))

    def _construct_hidden_layer(self):
        layers = self._params['hidden_layers']
        units = self._params['hidden_layer_units']
        for _ in range(layers):
            self.model.add(keras.layers.Dense(units, activation='relu'))

    def _construct_output_layer(self):
        self.model.add(keras.layers.Dense(CIRCLES_MAX, activation='softmax'))

    def _get_model_name(self):
        layers = self._params['hidden_layers']
        units = self._params['hidden_layer_units']
        return '{}.{}-{}'.format(MODEL_NAME_PREFIX, layers, units)

    def _get_loss(self):
        if self.__sparse:
            loss = 'sparse_categorical_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        return loss

    def _get_metrics(self):
        return ['accuracy']

    def _extract_date(self, data, test_data):
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
        return validation_data, x_train, y_train

    def _extract_verification_data(self, data):
        x, y_reg, y_cls = data

        if self.__sparse:
            y = y_reg
        else:
            y = y_cls
        return x, y_reg, y

    def __get_model_path(self):
        if self.model:
            name = self.model.name
        else:
            name = self._get_model_name()
        return os.path.join(MODEL_BASE_DIR, name), os.path.join(
            MODEL_BASE_DIR, name + '.old'
        )


class ClassificationModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def _get_model_name(self):
        layers = self._params['hidden_layers']
        units = self._params['hidden_layer_units']
        return '{}.cls.{}-{}'.format(MODEL_NAME_PREFIX, layers, units)


class RegressionModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def _get_model_name(self):
        layers = self._params['hidden_layers']
        units = self._params['hidden_layer_units']
        return '{}.reg.{}-{}'.format(MODEL_NAME_PREFIX, layers, units)

    def _get_loss(self):
        return 'mean_squared_error'

    def _get_metrics(self):
        return ['accuracy']

    def _construct_output_layer(self):
        self.model.add(keras.layers.Dense(1))

    def _extract_date(self, data, test_data):
        validation_data = None
        if test_data is not None:
            x_test, y_test, _ = test_data
            validation_data = (x_test, y_test)
        x_train, y_train, _ = data
        return validation_data, x_train, y_train

    def _extract_verification_data(self, data):
        x, y, _ = data
        return x, y, y


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
