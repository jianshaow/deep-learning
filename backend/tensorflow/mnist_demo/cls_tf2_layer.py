import tensorflow as tf
from keras.datasets import mnist
from common import layers, models
from common import vis_utils as vis


def run():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(32)

    model = models.SimpleSequential(
        [
            layers.SimpleFlatten(input_shape=(28, 28)),
            layers.SimpleDense(128, activation="relu"),
            layers.SimpleDense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        dataset,
        validation_data=(x_test, y_test),
        callbacks=[vis.matplotlib_callback()],
        epochs=10,
    )


if __name__ == "__main__":
    run()
