import os, keras, mnist_demo
from tkinter import N
from keras.datasets import mnist
from common import MODEL_BASE_DIR, vis_utils as vis
from mnist_demo import img_utils


MODEL_NAME = "mnist.cls.fc1-128"
MODEL_DIR = os.path.join(MODEL_BASE_DIR, MODEL_NAME)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_data, test_data = mnist_demo.dataset.prepare_data(
        (x_train, y_train), (x_test, y_test)
    )
    return train_data, test_data


def train(data, model=None, save_needed=False):
    train_data, test_data = data

    if model is None:
        model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_data,
        validation_data=test_data,
        callbacks=[vis.matplotlib_callback()],
        epochs=10,
    )

    if save_needed:
        model.save(MODEL_DIR)

    return model


def load_model():
    if os.path.exists(MODEL_DIR):
        return keras.models.load_model(MODEL_DIR, compile=compile)
    else:
        return None


def verify(model, data):
    image, _ = data
    preds = model.predict(image)
    img_utils.show_images(data, preds=preds)


if __name__ == "__main__":
    data = load_data()
    model = load_model()
    # model = train(data)
    _, test_data = data
    data = test_data.as_numpy_iterator().next()
    verify(model, data)
