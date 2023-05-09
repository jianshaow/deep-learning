import cc_model
import data_utils as utils

from circle_count import DEFAULT_MODEL_PARAMS, LEARNING_RATE

TRAIN_EPOCHS = 10
RERUN_EPOCHS = 20


def first_run(data_loader=utils.load_data, learning_rate=LEARNING_RATE, dry_run=False):
    model = __new_model()
    model.build()
    model.compile(learning_rate)
    model.train(data_loader(), epochs=TRAIN_EPOCHS)
    model.verify(utils.gen_sample_data(size=100))

    if not dry_run:
        model.save()

    return model


def re_run(
    data_loader=utils.load_data, learning_rate=LEARNING_RATE, epochs=RERUN_EPOCHS
):
    model = __new_model()
    model.load()
    model.compile(learning_rate)
    model.train(data_loader(), epochs=epochs)
    model.verify(utils.gen_sample_data(size=100))
    model.save(ask=True)

    return model


def demo_model(data_loader=lambda: utils.gen_sample_data(size=100)):
    model = __new_model()
    model.load()
    model.compile()
    model.verify(data_loader())


def __new_model(model_params=DEFAULT_MODEL_PARAMS):
    return cc_model.new_model('RegressionModel', model_params)


if __name__ == '__main__':
    # first_run()
    # first_run(dry_run=True)
    # first_run(learning_rate=0.0001)
    # first_run(dry_run=True, learning_rate=0.000001)
    re_run()
    # re_run(learning_rate=0.0001)
    # re_run(learning_rate=0.000001)
    # re_run(data_loader=utils.load_error_data, epochs=40)
    # re_run(data_loader=lambda: utils.load_error_data(error_gt=0.2), epochs=40)
    # re_run(data_loader=utils.load_error_data, learning_rate=0.0001)
    # re_run(data_loader=utils.load_error_data, learning_rate=0.000001)
    # demo_model()
    # demo_model(data_loader=lambda: utils.load_sample_data(size=1000))
    # demo_model(data_loader=lambda: utils.load_sample_error_data(size=1000))
