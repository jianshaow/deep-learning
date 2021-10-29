from common import img_utils as img
from common.img_utils import TRAIN_EPOCH

import models
import utils
from models import LEARNING_RATE, MODEL_PARAMS

DATA_CONFIG = img.data_config(6)


def first_run(model_params=MODEL_PARAMS, data=utils.prepare_data(), learning_rate=LEARNING_RATE, dry_run=False):
    x_train, y_train, validation_data = data

    model = models.Model(model_params)
    model.build()
    model.compile(learning_rate)
    model.train(x_train, y_train, validation_data=validation_data)
    model.verify(img.gen_circles_data(DATA_CONFIG, size=100))

    if not dry_run:
        model.save()

    return model


def re_run(model_params=MODEL_PARAMS, data=utils.prepare_data(), learning_rate=None, epochs=TRAIN_EPOCH):
    x_train, y_train, validation_data = data

    model = models.Model(model_params)
    model.load()

    if learning_rate is not None:
        model.compile(learning_rate)

    model.train(x_train, y_train, epochs=epochs,
                validation_data=validation_data)

    model.verify(img.gen_circles_data(DATA_CONFIG, size=100))

    model.save(ask=True)

    return model


def demo_model(model_params=MODEL_PARAMS, data=img.gen_circles_data(DATA_CONFIG, size=100)):
    model = models.Model(model_params)
    model.load()
    model.verify(data)


if __name__ == '__main__':
    # first_run()
    # first_run(learning_rate=0.00001)
    # first_run(dry_run=True)
    re_run()
    # re_run(data=prepare_error_data(), learning_rate=0.000001, epochs=10)
    # demo_model()
    # demo_model(data=load_sample_data(1000))
    # demo_model(data=load_sample_error_data(1000))
