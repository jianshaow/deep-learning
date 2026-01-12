import os
import sys

import numpy as np
import tensorflow as tf
from keras import Sequential, models

import common
from circle_count import img_utils


def deprocess_image(x):
    """
    Takes a tensor/numpy array and normalizes it for image display.
    """
    x = tf.cast(x, tf.float32)
    x -= tf.reduce_mean(x)
    x /= tf.math.reduce_std(x) + 1e-5
    x *= 0.1

    x += 0.5
    x = tf.clip_by_value(x, 0, 1)

    x *= 255  # type: ignore
    x = tf.clip_by_value(x, 0, 255).numpy().astype("uint8")  # type: ignore
    return x


if __name__ == "__main__":
    model_path = os.path.join(common.MODEL_BASE_DIR, "circle-count_ConvRegModel_fc1-128_conv2-64.h5")
    model = models.load_model(model_path, compile=False)
    if not isinstance(model, Sequential):
        raise ValueError("Loaded model is not a Keras Model instance.")
    model.summary()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = len(sys.argv) >= 2 and sys.argv[1] or "conv2d"

    target_layer = layer_dict[layer_name]
    activation_model = models.Model(inputs=model.inputs, outputs=target_layer.output)

    kept_filters = []
    img_width, img_height, channels = 100, 100, 1

    for filter_index in range(32):
        input_img_data = tf.random.uniform((1, img_width, img_height, channels))
        input_img_data = tf.Variable(input_img_data)

        for i in range(20):
            with tf.GradientTape() as tape:
                tape.watch(input_img_data)
                layer_output = activation_model(input_img_data)
                loss = tf.math.reduce_mean(
                    layer_output[:, :, :, filter_index], keepdims=True
                )

            grads = tape.gradient(loss, input_img_data)
            grads_norm = tf.math.sqrt(tf.reduce_mean(tf.math.square(grads))) + 1e-5
            grads /= grads_norm
            input_img_data.assign_add(grads)
            loss_value = loss.numpy()

            if loss_value <= 0.0:
                break

        if loss_value > 0:
            img = deprocess_image(input_img_data[0])  # type: ignore
            kept_filters.append((img, loss_value))

    # kept_filters.sort(key=lambda x: x[1], reverse=True)
    # kept_filters = kept_filters[:20]
    images = [item[0] for item in kept_filters]

    img_utils.show_images((images, np.arange(len(images))))
