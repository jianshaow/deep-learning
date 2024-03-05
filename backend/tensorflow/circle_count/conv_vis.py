import sys, numpy as np
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import circle_count as cc
from circle_count import cc_model, img_utils


def deprocess_image(x):
    x -= x.mean()
    x /= x.std() + 1e-5
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


if __name__ == "__main__":
    disable_eager_execution()
    params = cc.CONV_REG_MODEL_PARAMS
    model = cc_model.load_model(params).model
    model.summary()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = len(sys.argv) >= 2 and sys.argv[1] or "conv2d"
    input_img = model.input

    kept_filters = []
    for filter_index in range(32):
        layer_output = layer_dict[layer_name].output

        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, input_img)[0]
        grads /= K.sqrt(K.mean(K.square(grads))) + 1e-5

        iterate = K.function([input_img], [loss, grads])

        input_img_data = np.random.random((1, 100, 100, 1))

        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value
            if loss_value <= 0.0:
                break

        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))

    # kept_filters.sort(key=lambda x: x[1], reverse=True)
    # kept_filters = kept_filters[:20]
    images = [item[0] for item in kept_filters]

    img_utils.show_images((images, np.arange(len(images))))
