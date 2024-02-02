import numpy as np
import tensorflow as tf
from imageio import imsave
from PIL import Image
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import circle_count as cc
from circle_count import data_utils as utils, cc_model


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
    data, _ = utils.gen_sample_data(get_config=cc.data_config((6, 6), (6, 7)), size=1)
    params = cc.CONV_REG_MODEL_PARAMS
    model = cc_model.load_model(params).model
    model.summary()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    print(layer_dict)

    layer_name = "conv2d"
    input_img = model.input

    filter_index = 0
    layer_output = layer_dict[layer_name].output

    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]
    grads /= K.sqrt(K.mean(K.square(grads))) + 1e-5

    iterate = K.function([input_img], [loss, grads])

    input_img_data = np.random.random((1, 100, 100, 1))

    kept_filters = []
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value

        print("Current loss value:", loss_value)
        if loss_value <= 0.0:
            break

    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))

    temp = np.zeros((100, 100, 3))
    temp[:] = img
    img = Image.fromarray((temp).astype(np.uint8))
    imsave("%s_filter_%d.png" % (layer_name, filter_index), img)
