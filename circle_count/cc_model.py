import os

import keras

import circle_count as cc
from circle_count import img_utils as img
from circle_count import model_store as store
from common import MODEL_BASE_DIR
from common import vis_utils as vis

MODEL_NAME_PREFIX = "circle-count"

TRAIN_EPOCHS = 10


class Model:
    def __init__(self, params):
        self.__compiled = False
        self._params = params
        self.model: keras.Sequential | None = None
        self.built = False

    def build(self):
        if self.built:
            raise RuntimeError("model is initialized")

        self._construct_model()
        self._construct_input_layer()
        self._construct_fc_layer()
        self._construct_output_layer()

        if self.model:
            self.model.summary()
            self.built = True

    def load(self, compile_needed=False):
        if self.model is not None:
            raise RuntimeError("model is initialized")

        model_path = self.__get_model_path()
        self.model = store.load(model_path, compile_needed=compile_needed)
        print(model_path, "loaded")
        self.model.summary()
        self.__compiled = compile_needed

    def save(self, ask=False):
        if self.model is None:
            raise RuntimeError("model is not initialized, call build or load method")

        if ask:
            save = input('save model ["{}"]? (y|n): '.format(self.model.name))
            if save != "y":
                print("model [{}] not saved".format(self.model.name))
                return

        model_path = self.__get_model_path()
        store.save(self.model, model_path)
        print("model [" + self.model.name + "] saved")

    def show(self):
        if self.model is None:
            raise RuntimeError("model is not initialized, call build or load method")

        vis.build_model_figure(self.model)
        vis.show_all()

    def compile(self, learning_rate=cc.LEARNING_RATE):
        if self.model is None:
            raise RuntimeError("model is not initialized, call build or load method")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=self._get_loss(),
            metrics=self._get_metrics(),
        )
        self.__compiled = True

    def train(self, data, epochs=TRAIN_EPOCHS, test_data=None):
        if self.model is None:
            raise RuntimeError("model is not initialized, call build or load method")

        if not self.__compiled:
            raise RuntimeError("model is not compiled yet, call compile first")

        train_data, test_data = cc.dataset.prepare_data(data, test_data)

        self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=test_data,
            callbacks=[
                vis.matplotlib_callback(),
                vis.tensorboard_callback("circle_count"),
            ],
        )

    def predict(self, x):
        if self.model is None:
            raise RuntimeError(
                "model is not initialized, call build or load method first"
            )

        x, _ = self._pre_process(x, None)
        prediction = self.model.predict(x)

        return prediction

    def evaluate(self, data):
        if self.model is None:
            raise RuntimeError(
                "model is not initialized, call build or load method first"
            )

        x, y = data
        x, y = self._pre_process(x, y)

        return self.model.evaluate(x, y)

    def verify(self, data):
        if self.model is None:
            raise RuntimeError(
                "model is not initialized, call build or load method first"
            )

        x, y = data
        x, y = self._pre_process(x, y)

        if self.__compiled:
            evaluation = self.model.evaluate(x, y)
            print("evaluation: ", evaluation)

        preds = self.model.predict(x)
        img.show_images(data, preds, title="predict result")

    def _construct_model(self):
        self.model = keras.Sequential(name=self._get_model_name())

    def _construct_input_layer(self):
        if self.model is None:
            raise RuntimeError(
                "model is not initialized, call build or load method first"
            )

        input_shape = self._params["input_shape"]
        self.model.add(keras.layers.Flatten(input_shape=input_shape))

    def _construct_fc_layer(self):
        if self.model is None:
            raise RuntimeError("model is not initialized, call build or load method")

        layers = self._params["fc_layers"]
        units = self._params["fc_layers_units"]
        for _ in range(layers):
            self.model.add(keras.layers.Dense(units, activation="relu"))

    def _construct_output_layer(self):
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not have a `_construct_output_layer()` "
            "method implemented."
        )

    def _get_model_name(self):
        layers = self._params["fc_layers"]
        units = self._params["fc_layers_units"]
        return "{}_{}_fc{}-{}".format(
            MODEL_NAME_PREFIX, self.__class__.__name__, layers, units
        )

    def _get_loss(self):
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not have a `_construct_output_layer()` "
            "method implemented."
        )

    def _get_metrics(self):
        return ["accuracy"]

    def _pre_process(self, x, y):
        return cc.dataset.pre_process(x, y)

    def _post_process(self, data):
        return data

    def __get_model_path(self):
        if self.model:
            name = self.model.name
        else:
            name = self._get_model_name()
        return os.path.join(MODEL_BASE_DIR, name)


class ClassificationModel(Model):

    def _get_loss(self):
        return "sparse_categorical_crossentropy"

    def _construct_output_layer(self):
        if self.model:
            output_units = self._params["output_units"]
            self.model.add(keras.layers.Dense(output_units, activation="softmax"))


class RegressionModel(Model):

    def _get_loss(self):
        return "mean_squared_error"

    def _construct_output_layer(self):
        if self.model is None:
            raise RuntimeError("model is not initialized, call build or load method")

        self.model.add(keras.layers.Dense(1))


class ConvModel(Model):

    def _get_model_name(self):
        layers = self._params["conv_layers"]
        filters = self._params["conv_filters"]
        return "{}_conv{}-{}".format(super()._get_model_name(), layers, filters)

    def _construct_input_layer(self):
        if self.model is None:
            raise RuntimeError("model is not initialized, call build or load method")

        input_shape = self._params["input_shape"]
        self.model.add(
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape)
        )
        layers = self._params["conv_layers"]
        filters = self._params["conv_filters"]
        for _ in range(layers):
            self.model.add(keras.layers.Conv2D(filters, (3, 3), activation="relu"))
            self.model.add(keras.layers.MaxPooling2D())
        self.model.add(keras.layers.Flatten())

    def _get_loss(self):
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not have a `_construct_output_layer()` "
            "method implemented."
        )


class ConvRegModel(RegressionModel, ConvModel):
    pass


class ConvClsModel(ClassificationModel, ConvModel):
    pass


def new_model(params):
    model_type = params["model_type"]
    if not model_type:
        model_type = RegressionModel
    if isinstance(model_type, str):
        model_class = globals()[model_type]
    else:
        model_class = model_type
    if model_class and issubclass(model_class, Model):
        return model_class(params)
    else:
        raise ValueError("no such model %s" % model_type)


def load_model(params, compile_needed=False):
    model_type = params["model_type"]
    if isinstance(model_type, str):
        model_class = globals()[model_type]
    else:
        model_class = model_type
    if model_class and issubclass(model_class, Model):
        model = model_class(params)
        model.load(compile_needed=compile_needed)
        return model
    else:
        raise RuntimeError("no such model %s" % model_type)


def __main():
    import circle_count.data_utils as dutils

    # data = dutils.gen_sample_data(size=20)
    data = dutils.gen_sample_data(get_config=cc.data_config((6, 6), (6, 7)), size=20)
    # data = dutils.load_data()
    # data = dutils.load_error_data()
    # data = dutils.load_error_data(error_gt=0.2)

    # params = cc.REG_MODEL_PARAMS
    # params = cc.CLS_MODEL_PARAMS
    params = cc.CONV_REG_MODEL_PARAMS
    # params = cc.CONV_CLS_MODEL_PARAMS
    model = load_model(params)
    model.show()
    model.verify(data)


if __name__ == "__main__":
    __main()
