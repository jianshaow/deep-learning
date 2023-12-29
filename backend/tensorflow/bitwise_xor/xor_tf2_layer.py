import keras
import tensorflow as tf
import bitwise_xor.data_utils as utils
from common import layers, models
from common import vis_utils as vis
from bitwise_xor.data_utils import SEQUENCE_SIZE, TRAIN_EPOCH


def build_model():
    model = models.SimpleSequential(
        [
            layers.SimpleFlatten(input_shape=(2, SEQUENCE_SIZE)),
            layers.SimpleDense(64, activation=keras.activations.relu),
            layers.SimpleDense(64, activation=keras.activations.relu),
            layers.SimpleDense(64, activation=keras.activations.relu),
            layers.SimpleDense(SEQUENCE_SIZE, activation=keras.activations.sigmoid),
        ]
    )

    return model


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )


def train_model(model, dataset, test_data):
    model.fit(
        dataset=dataset,
        epochs=TRAIN_EPOCH,
        validation_data=test_data,
        callbacks=[vis.matplotlib_callback()],
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
    dataset = tf.data.Dataset.from_tensor_slices(utils.gen_xor_train_data())
    dataset = dataset.batch(32)
    test_data = utils.gen_xor_test_data()

    model = build_model()
    compile_model(model)
    train_model(model, dataset, test_data)
    verify_model(model)


def show():
    model = build_model()
    vis.build_model_figure(model)
    vis.show_all()


if __name__ == "__main__":
    show()
    # run()
