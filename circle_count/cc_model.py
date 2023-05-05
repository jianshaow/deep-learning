import os
import shutil

import img_utils as img
import tensorflow as tf
from tensorflow import keras

from circle_count import DEFAULT_MODEL_PARAMS, LEARNING_RATE
from common import data_dir
from common import vis_utils as vis

MODEL_NAME_PREFIX = 'circle_count'
MODEL_BASE_DIR = os.path.join(data_dir, 'model')

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

        size = len(data[0])
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(size, reshuffle_each_iteration=True)
        dataset = dataset.map(self._pre_process)

        if test_data is None:
            train_num = round(size * 0.9)
            train_data = dataset.take(train_num).batch(32)
            test_data = dataset.skip(train_num).batch(32)
        else:
            train_data = dataset.batch(32)
            test_size = len(test_data[0])
            test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
            test_dataset = test_dataset.shuffle(
                test_size, reshuffle_each_iteration=True
            )
            test_data = test_dataset.map(self._pre_process).batch(32)

        train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_data = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.fit(
            train_data,
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

        x, _ = self._pre_process(x, None)
        prediction = self.model.predict(x)

        return prediction

    def evaluate(self, data):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method first')

        x, y = data
        x, y = self._pre_process(x, y)

        return self.model.evaluate(x, y)

    def verify(self, data):
        if self.model is None:
            raise Exception('model is not initialized, call build or load method first')

        if not self.__compiled:
            raise Exception('model is not compiled yet, call compile first')

        x, y = data
        x, y = self._pre_process(x, y)

        preds = self.model.predict(x)
        img.show_images(data, preds, title='predict result')

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

    def _pre_process(self, x, y):
        x = tf.cast(x, tf.float32) / 255.0
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
    import data_utils as dutils

    data = dutils.gen_sample_data(size=100)
    # data = dutils.load_data()
    # data = dutils.load_error_data()

    model = RegressionModel(DEFAULT_MODEL_PARAMS)
    model.load(compile=True)
    model.verify(data)
