import cc_model
import data_utils as utils
from cc_model import LEARNING_RATE, MODEL_PARAMS

TRAIN_EPOCHS = 10
RERUN_EPOCHS = 20


def first_run(data=utils.load_data(), learning_rate=LEARNING_RATE, dry_run=False):
    model = __new_model()
    model.build()
    model.compile(learning_rate)
    model.train(data, epochs=TRAIN_EPOCHS)
    model.verify(utils.gen_sample_data(size=100))

    if not dry_run:
        model.save()

    return model


def re_run(data=utils.load_data(), learning_rate=LEARNING_RATE, epochs=RERUN_EPOCHS):
    model = __new_model()
    model.load()
    model.compile(learning_rate)
    model.train(data, epochs=epochs)
    model.verify(utils.gen_sample_data(size=100))
    model.save(ask=True)

    return model


def demo_model(data=utils.gen_sample_data(size=100)):
    model = __new_model()
    model.load()
    model.compile()
    model.verify(data)


def __new_model():
    return cc_model.new_model('RegressionModel', MODEL_PARAMS)


if __name__ == '__main__':
    # first_run()
    # first_run(dry_run=True)
    # first_run(learning_rate=0.0001)
    # first_run(dry_run=True, learning_rate=0.000001)
    re_run()
    # re_run(learning_rate=0.0001)
    # re_run(learning_rate=0.000001)
    # re_run(data=utils.load_error_data())
    # re_run(data=utils.load_error_data(), learning_rate=0.0001)
    # re_run(data=utils.load_error_data(), learning_rate=0.000001)
    # demo_model()
    # demo_model(data=utils.load_sample_data(size=1000))
    # demo_model(data=utils.load_sample_error_data(size=1000))
