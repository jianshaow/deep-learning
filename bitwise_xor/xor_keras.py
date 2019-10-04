from tensorflow import keras

from common import data_utils as utils
from common import vis_utils as vis
from common.data_utils import SEQUENCE_SIZE, TRAIN_EPOCH


def run():
    train_data, train_labels = utils.gen_xor_train_data()
    test_data = utils.gen_xor_test_data()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(2, SEQUENCE_SIZE)),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(SEQUENCE_SIZE, activation=keras.activations.sigmoid)
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy()])

    callback = vis.VisualizationCallback(
        show_model=True, show_metrics=True, dynamic_plot=True)
    model.fit(train_data, train_labels,
              validation_data=test_data,
              epochs=TRAIN_EPOCH,
              callbacks=[callback])

    example_data = utils.random_seq_pairs(1)
    example_result = model.predict(example_data)
    vis.build_multi_bar_figure(['seq1', 'seq2', 'xor'],
                               [example_data[0][0], example_data[0][1], example_result[0]])
    vis.show_all()


if __name__ == '__main__':
    run()
