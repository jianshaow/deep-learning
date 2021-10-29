import cc_model
import data_utils as utils
from cc_model import LEARNING_RATE, MODEL_PARAMS, TRAIN_EPOCH


def first_run(model_params=MODEL_PARAMS, data=utils.prepare_data(), learning_rate=LEARNING_RATE, dry_run=False):
    x_train, y_train, validation_data = data

    model = cc_model.Model(model_params)
    model.build()
    model.compile(learning_rate)
    model.train(x_train, y_train, validation_data=validation_data)
    model.verify(utils.gen_circles_data(size=100))

    if not dry_run:
        model.save()

    return model


def re_run(model_params=MODEL_PARAMS, data=utils.prepare_data(), learning_rate=None, epochs=TRAIN_EPOCH):
    x_train, y_train, validation_data = data

    model = cc_model.Model(model_params)
    model.load()

    if learning_rate is not None:
        model.compile(learning_rate)

    model.train(x_train, y_train, epochs=epochs,
                validation_data=validation_data)
    model.verify(utils.gen_circles_data(size=100))
    model.save(ask=True)

    return model


def demo_model(model_params=MODEL_PARAMS, data=utils.gen_circles_data(size=100)):
    model = cc_model.Model(model_params)
    model.load()
    model.verify(data)


if __name__ == '__main__':
    # first_run()
    # first_run(learning_rate=0.00001)
    # first_run(dry_run=True)
    re_run()
    # re_run(data=utils.prepare_error_data(), learning_rate=0.0001, epochs=100)
    # demo_model()
    # demo_model(data=utils.load_sample_data(1000))
    # demo_model(data=utils.load_sample_error_data(1000))
