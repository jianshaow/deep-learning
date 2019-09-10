from tensorflow import keras

from common import vis

if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callback = vis.VisualizationCallback(show_model=True, runtime_plot=True)
    model.fit(train_images, train_labels,
              validation_data=(test_images, test_labels),
              callbacks=[callback],
              epochs=10)
