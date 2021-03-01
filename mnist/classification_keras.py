from tensorflow import keras
from tensorflow.keras.datasets import mnist

from common import vis_utils as vis


def run():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callback = vis.VisualizationCallback(
        show_model=True, show_metrics=True, dynamic_plot=True)
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              callbacks=[callback],
              epochs=10)


if __name__ == '__main__':
    run()
