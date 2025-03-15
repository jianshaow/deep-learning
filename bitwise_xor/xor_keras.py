import keras

import bitwise_xor.data_utils as utils
from bitwise_xor.data_utils import SEQUENCE_SIZE, TRAIN_EPOCH
from common import vis_utils as vis


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Input((2, SEQUENCE_SIZE)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation=keras.activations.relu),
            keras.layers.Dense(64, activation=keras.activations.relu),
            keras.layers.Dense(64, activation=keras.activations.relu),
            keras.layers.Dense(SEQUENCE_SIZE, activation=keras.activations.sigmoid),
        ]
    )
    return model


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )


def train_model(model, train_data, train_labels, test_data):
    model.fit(
        train_data,
        train_labels,
        validation_data=test_data,
        epochs=TRAIN_EPOCH,
        callbacks=[vis.matplotlib_callback(), vis.tensorboard_callback("bitwise_xor")],
    )


def verify_model(model):
    example_data = utils.random_seq_pairs(1)
    example_result = model.predict(example_data)
    vis.build_multi_bar_figure(
        ["seq1", "seq2", "xor"],
        [example_data[0][0], example_data[0][1], example_result[0]],
    )
    vis.show_all()


def run():
    train_data, train_labels = utils.gen_xor_train_data()
    test_data = utils.gen_xor_test_data()

    model = build_model()
    compile_model(model)
    train_model(model, train_data, train_labels, test_data)
    verify_model(model)


def show():
    model = build_model()
    vis.build_model_figure(model)
    vis.show_all()


if __name__ == "__main__":
    show()
    # run()
