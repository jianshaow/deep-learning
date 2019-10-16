from tensorflow import keras

from common import vis_utils as vis


def run():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callback = vis.VisualizationCallback(
        show_model=True, show_metrics=True, dynamic_plot=True)
    model.fit(train_images, train_labels,
              validation_data=(test_images, test_labels),
              callbacks=[callback],
              epochs=10)


if __name__ == '__main__':
    run()
