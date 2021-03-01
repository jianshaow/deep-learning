from tensorflow import keras

from common import img_utils as utils
from common import vis_utils as vis
from common.img_utils import TRAIN_EPOCH, CIRCLES_MAX


def run():
    (train_data, _, train_labels), (test_data, _, test_labels) = utils.load_data()
    train_data, test_data = train_data/255.0, test_data/255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 100)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(CIRCLES_MAX, 'sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    callback = vis.VisualizationCallback(
        show_model=True, show_metrics=True, dynamic_plot=True)
    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              epochs=TRAIN_EPOCH,
              callbacks=[callback])


if __name__ == '__main__':
    run()
