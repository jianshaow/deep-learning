import circle_count as cc, data_utils as utils, cc_model


TRAIN_EPOCHS = 10
RERUN_EPOCHS = 20


def first_run(
    model_params=cc.REG_MODEL_PARAMS,
    data_loader=utils.load_data,
    learning_rate=cc.LEARNING_RATE,
    epochs=TRAIN_EPOCHS,
    dry_run=False,
):
    model = __new_model(model_params)
    model.build()
    model.compile(learning_rate)
    model.train(data_loader(), epochs=epochs)
    model.verify(utils.gen_sample_data(size=100))

    if not dry_run:
        model.save()

    return model


def re_run(
    model_params=cc.REG_MODEL_PARAMS,
    data_loader=utils.load_data,
    learning_rate=cc.LEARNING_RATE,
    epochs=RERUN_EPOCHS,
):
    model = __new_model(model_params)
    model.load()
    model.compile(learning_rate)
    model.train(data_loader(), epochs=epochs)
    model.verify(utils.gen_sample_data(size=100))
    model.save(ask=True)

    return model


def demo_model(
    model_params=cc.REG_MODEL_PARAMS,
    data_loader=lambda: utils.gen_sample_data(size=100),
):
    model = __new_model(model_params)
    model.load()
    model.compile()
    model.verify(data_loader())


def __new_model(model_params):
    return cc_model.new_model(model_params)


if __name__ == "__main__":
    # first_run()
    # first_run(model_params=cc.CLS_MODEL_PARAMS)
    # first_run(model_params=cc.CONV_REG_MODEL_PARAMS)
    # first_run(model_params=cc.CONV_CLS_MODEL_PARAMS)
    # first_run(dry_run=True, epochs=1)
    # first_run(learning_rate=0.0001)
    # first_run(dry_run=True, learning_rate=0.000001)
    # re_run()
    # re_run(model_params=cc.CONV_REG_MODEL_PARAMS, epochs=10)
    # re_run(learning_rate=0.000001)
    # re_run(data_loader=utils.load_error_data, epochs=40)
    # re_run(data_loader=lambda: utils.load_error_data(error_gt=0.2), epochs=40)
    # re_run(data_loader=utils.load_error_data, learning_rate=0.000001)
    demo_model()
    # demo_model(model_params=cc.CONV_REG_MODEL_PARAMS)
    # demo_model(data_loader=lambda: utils.load_sample_data(size=1000))
    # demo_model(data_loader=lambda: utils.load_sample_error_data(size=1000))
    pass
