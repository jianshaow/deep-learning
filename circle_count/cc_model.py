import os
import shutil

from tensorflow import keras

from common import img_utils as img
from common import vis_utils as vis

MODEL_NAME_PREFIX = 'circle_count'
MODEL_BASE_DIR = os.path.join(os.path.expanduser('~'), '.model')

MODEL_PARAMS = {
    'input_shape': (100, 100),
    'hidden_layers': 3,
    'hidden_layer_units': 64,
    'output_units': 6,
}
LEARNING_RATE = 0.00001

TRAIN_EPOCHS = 10


def new_model(type, params):
    if isinstance(type, str):
        model_class = globals()[type]
    else:
        model_class = type
    if model_class and issubclass(model_class, Model):
        return model_class(params)
    else:
        raise Exception('no such model' + type)


def load_model(type, params, compile=False):
    if isinstance(type, str):
        model_class = globals()[type]
    else:
        model_class = type
    if model_class and issubclass(model_class, Model):
        model = model_class(params)
        model.load(compile=compile)
        return model
    else:
        raise Exception('no such model' + type)


class Model:
    def __init__(self, params):
        self.__compiled = False
        self._params = params
        self.model = None

    def build(self):
        if self.model is not None:
            raise Exception('model is initialized')

        self._construct_model()
        self._construct_input_layer()
        self._construct_hidden_layer()
        self._construct_output_layer()
        self.model.summary()

    def load(self, compile=False):
        if self.model is not None:
            raise Exception('model is initialized')

        model_path, _ = self.__get_model_path()
        self.model = keras.models.load_model(model_path, compile=compile)
        self.model.summary()
        self.__compiled = compile

    def compile(self, learning_rate=LEARNING_RATE):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method')

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=self._get_loss(),
            metrics=self._get_metrics(),
        )
        self.__compiled = True

    def train(self, data, epochs=TRAIN_EPOCHS, test_data=None):
        if not self.__compiled:
            raise Exception('model is not compiled yet, call compile first')

        x_train, y_train = self._pre_process(data)
        test_data = self._pre_process(test_data)

        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=test_data,
            callbacks=[
                vis.matplotlib_callback(),
                vis.tensorboard_callback('circle_count'),
            ],
        )

    def predict(self, x):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method first')

        x, _ = self._pre_process((x, None))
        prediction = self.model.predict(x)

        return prediction

    def evaluate(self, data):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method first')

        x, y = self._pre_process(data)

        return self.model.evaluate(x, y)

    def verify(self, data):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method first')

        if not self.__compiled:
            raise Exception('model is not compiled yet, call compile first')

        x, y = self._pre_process(data)

        predictions = self.model.predict(x)
        img.show_images(data, predictions, title='predict result')

        evaluation = self.model.evaluate(x, y)
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
        self.model = keras.Sequential(name=self._get_model_name())

    def _construct_input_layer(self):
        input_shape = self._params['input_shape']
        self.model.add(keras.layers.Flatten(input_shape=input_shape))

    def _construct_hidden_layer(self):
        layers = self._params['hidden_layers']
        units = self._params['hidden_layer_units']
        for _ in range(layers):
            self.model.add(keras.layers.Dense(units, activation='relu'))

    def _construct_output_layer(self):
        output_units = self._params['output_units']
        self.model.add(keras.layers.Dense(output_units, activation='softmax'))

    def _get_model_name(self):
        layers = self._params['hidden_layers']
        units = self._params['hidden_layer_units']
        return '{}.{}-{}'.format(MODEL_NAME_PREFIX, layers, units)

    def _get_loss(self):
        return 'sparse_categorical_crossentropy'

    def _get_metrics(self):
        return ['accuracy']

    def _pre_process(self, data):
        x, y = data
        if x is not None:
            x = x / 255.0
        return (x, y)

    def _post_process(self, data):
        return data

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
        output_units = self._params['output_units']
        return '{}.cls.{}-{}.{}'.format(MODEL_NAME_PREFIX, layers, units, output_units)


class RegressionModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def _get_model_name(self):
        layers = self._params['hidden_layers']
        units = self._params['hidden_layer_units']
        output_units = self._params['output_units']
        return '{}.reg.{}-{}.{}'.format(MODEL_NAME_PREFIX, layers, units, output_units)

    def _get_loss(self):
        return 'mean_squared_error'

    def _construct_hidden_layer(self):
        layers = self._params['hidden_layers']
        units = self._params['hidden_layer_units']
        for _ in range(layers):
            self.model.add(keras.layers.Dense(units, activation='relu'))

    def _construct_output_layer(self):
        self.model.add(keras.layers.Dense(1))


if __name__ == '__main__':
    images, nums = img.zero_data(20)

    def handle(i, image, num):
        images[i] = image
        nums[i] = num

    img.random_circles_images(handle, size=20)

    model = RegressionModel(MODEL_PARAMS)
    model.load()
    predictions = model.predict(images)
    img.show_images((images, nums), predictions)
