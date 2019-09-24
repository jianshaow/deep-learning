import tensorflow as tf
from tensorflow import keras

import xor_util as util
from xor_util import SEQUENCE_SIZE, TRAINING_EPOCH
from common import models
from common import vis_utils as vis

if __name__ == '__main__':
    dataset = tf.data.Dataset.from_tensor_slices(util.load_training_data())
    dataset = dataset.batch(32)  # .take(1)

    model = models.SimpleModel([
        keras.layers.Flatten(input_shape=(2, SEQUENCE_SIZE)),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(SEQUENCE_SIZE, activation=keras.activations.sigmoid)])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy()])

    callback = vis.VisualizationCallback(show_model=True, runtime_plot=True)
    model.fit(dataset=dataset,
              epochs=TRAINING_EPOCH,
              callbacks=[callback]
              )
