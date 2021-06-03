import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import Sequential


def setup_model() -> Sequential:
    # Set up a simple CNN
    seq_model = tf.keras.models.Sequential()
    seq_model.add(layers.Conv2D(filters=32, kernel_size=(3, 3,), activation="relu", input_shape=(28, 28, 1,)))
    seq_model.add(layers.MaxPooling2D(pool_size=(2, 2,)))
    seq_model.add(layers.Conv2D(64, (5, 5,), activation="relu"))
    seq_model.add(layers.MaxPooling2D((2, 2,)))
    seq_model.add(layers.Conv2D(64, (3, 3,)))
    seq_model.add(layers.Dropout(0.5))

    seq_model.add(layers.Flatten())
    seq_model.add(layers.Dense(32, activation="relu"))
    seq_model.add(layers.Dense(10, activation="softmax"))

    print(seq_model.summary())

    seq_model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.categorical_crossentropy
    )

    return seq_model
