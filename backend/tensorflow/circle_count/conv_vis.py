from tensorflow.python.framework.ops import disable_eager_execution
from keras import backend as K
import circle_count as cc
from circle_count import data_utils as utils, cc_model

if __name__ == "__main__":
    disable_eager_execution()
    data = utils.gen_sample_data(get_config=cc.data_config((6, 6), (6, 7)), size=20)
    params = cc.CONV_REG_MODEL_PARAMS
    model = cc_model.load_model(params)
    # model.show()
    layer_dict = dict([(layer.name, layer) for layer in model.model.layers])
    print(layer_dict)

    layer_name = 'conv2d'
    filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, data[0][0])[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([data[0][0]], [loss, grads])