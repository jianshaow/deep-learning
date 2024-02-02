import tensorflow as tf


def prepare_data(train_data, test_data):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.shuffle(
        train_dataset.cardinality(), reshuffle_each_iteration=True
    )
    train_data = train_dataset.batch(32)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_dataset = test_dataset.shuffle(
        test_dataset.cardinality(), reshuffle_each_iteration=True
    )
    test_data = test_dataset.batch(32)

    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_data = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_data.take(1)

    return train_data, test_data


if __name__ == "__main__":
    from keras.datasets import mnist

    train_data, test_data = mnist.load_data()
    _, test_data = prepare_data(train_data, test_data)
    one = test_data.take(1)
    print(one.as_numpy_iterator().next()[1])
