from tensorflow import keras

from common import img_utils as utils
from common import vis_utils as vis
from common.img_utils import TRAIN_EPOCH


def run():
    train_data, train_labels = utils.load_circle_count_train_data()
    test_data, test_labels = utils.load_circle_count_test_data()
    train_data, test_data = train_data/255.0, test_data/255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(400, 400)),
        keras.layers.Dense(64, activation='relu', kernel_initializer='zeros'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse'])

    callback = vis.VisualizationCallback(
        show_model=True, show_metrics=True, dynamic_plot=True)
    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              epochs=TRAIN_EPOCH,
              callbacks=[callback])


if __name__ == '__main__':
    run()
