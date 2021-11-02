import tensorflow as tf
from common import data_utils as utils
from common import layers, models
from common import vis_utils as vis
from common.data_utils import SEQUENCE_SIZE, TRAIN_EPOCH
from tensorflow import keras


def run():
    dataset = tf.data.Dataset.from_tensor_slices(utils.gen_xor_train_data())
    dataset = dataset.batch(32)
    test_seq_pairs, test_labels = utils.gen_xor_test_data()

    model = models.SimpleSequential([
        layers.SimpleFlatten(input_shape=(2, SEQUENCE_SIZE)),
        layers.SimpleDense(64, activation=keras.activations.relu),
        layers.SimpleDense(64, activation=keras.activations.relu),
        layers.SimpleDense(64, activation=keras.activations.relu),
        layers.SimpleDense(SEQUENCE_SIZE, activation=keras.activations.sigmoid)])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy()])

    callback = vis.matplotlib_callback()
    model.fit(dataset=dataset,
              epochs=TRAIN_EPOCH,
              validation_data=(test_seq_pairs, test_labels),
              callbacks=[callback])

    example_data = utils.random_seq_pairs(1)
    example_result = model.predict(example_data)
    vis.build_multi_bar_figure(['seq1', 'seq2', 'xor'],
                               [example_data[0][0], example_data[0][1], example_result[0]])
    vis.show_all()


if __name__ == '__main__':
    run()
