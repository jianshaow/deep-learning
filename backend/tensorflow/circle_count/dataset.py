import tensorflow as tf


def prepare_data(data):
    size = len(data[0])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(size, reshuffle_each_iteration=True)
    dataset = dataset.map(pre_process)

    if test_data is None:
        train_num = round(size * 0.9)
        train_data = dataset.take(train_num).batch(32)
        test_data = dataset.skip(train_num).batch(32)
    else:
        train_data = dataset.batch(32)
        test_size = len(test_data[0])
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_dataset = test_dataset.shuffle(test_size, reshuffle_each_iteration=True)
        test_data = test_dataset.map(pre_process).batch(32)

    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_data = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, test_data


def pre_process(x, y):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.cast(x, tf.float32) / 255.0
    return (x, y)
