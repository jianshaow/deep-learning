import keras, mnist_demo
from keras.datasets import mnist
from common import vis_utils as vis


def run():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_data, test_data = mnist_demo.dataset.prepare_data(
        (x_train, y_train), (x_test, y_test)
    )

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        train_data,
        validation_data=test_data,
        callbacks=[vis.matplotlib_callback()],
        epochs=10,
    )


if __name__ == "__main__":
    run()
