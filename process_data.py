import tensorflow as tf
import numpy as np


def process_data(feature: np.ndarray, label: np.ndarray = None):
    n_obs = feature.shape[0]
    feature = feature.reshape((n_obs, 28, 28, 1,))
    feature = feature.astype("float32")/255
    if label is not None:
        label = tf.keras.utils.to_categorical(label)

    return feature, label,
