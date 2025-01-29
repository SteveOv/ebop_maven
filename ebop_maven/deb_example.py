"""
Details of the deb_example protobuf and support functions
"""
from typing import Dict, List

import numpy as np
import tensorflow as tf

# We can store multiple configuration of the mags feature but we can publish only one.
stored_mags_features = {
    # "mags_1024": 1024,
    # "mags_2048": 2048,
    "mags_4096": 4096,
}
default_mags_key = "mags_4096"                                  # pylint: disable=invalid-name
default_mags_bins = stored_mags_features[default_mags_key]      # pylint: disable=invalid-name
default_mags_wrap_phase = 1.0                                   # pylint: disable=invalid-name


# Python 3.7+ language spec ensures dictionary order is preserved
# The full set of stored labels and any scaling applied to them when read
labels_and_scales = {
    "rA_plus_rB": 1.,
    "k": 1.,
    "J": 1.,
    "qphot": 1.,
    "ecosw": 1.,
    "esinw": 1.,
    "inc": 0.01,
    "bP": 1.,
    "L3": 1.
}

# Extra features other than the light-curve used for predictions
extra_features_and_defaults = {
    "phiS": 0.5,        # Phase of secondary eclipse; correlated with ecosw
    "dS_over_dP": 1.0,  # Ratio of eclipse durations; correlated with esinw
}

# The deb Example protobuf description used by tf.Datasets
description = {
    # General purpose identifier
    "id": tf.io.FixedLenFeature([], tf.string, default_value=""),

    # Every label is expected to have a single float value
    **{ k: tf.io.FixedLenFeature([], tf.float32, default_value=0.)
                        for k in labels_and_scales },

    # Complex features: multiple configs of the phase folded light-curve mags
    **{ k: tf.io.FixedLenFeature((bins,), tf.float32, [0.] * bins)
                        for k, bins in stored_mags_features.items() },

    # Extra features are expected to have a single float value each
    **{ k: tf.io.FixedLenFeature([], tf.float32, default_value=default_val)
                        for k, default_val in extra_features_and_defaults.items() },
}

def create_mags_key(mags_bins: int) -> str:
    """ Helper function to format a key to the stored_mags_features dict """
    return f"mags_{int(mags_bins)}"

def get_all_extra_feature_names() -> List[str]:
    """ A list of the all the known extra feature names in the order they're stored. """
    return [*extra_features_and_defaults]

def get_all_label_names() -> List[str]:
    """ A list of the all the known label names in the order they're stored. """
    return [*labels_and_scales]

def serialize(identifier: str,
              labels: Dict[str, float],
              mags_features: Dict[str, List[float]],
              extra_features: Dict[str, float]) -> bytearray:
    """
    Encode the requested test instance data to a TF Example protobuf.

    :identifier: unique id of the instance
    :labels: the training labels
    :mags_features: the various configurations of phase folded time series magnitude data
    :extra_features: single value features associated with the light-curve
    :returns: a bytearray containing the serialized data
    """
    if default_mags_key not in mags_features:
        raise ValueError(f"The published {default_mags_key} not found in mags_features.")

    features = {
        "id": _to_bytes_feature(identifier),

        # Labels - will raise KeyError if an expected label is not given
        **{ k: _to_float_feature(labels[k]) for k in labels_and_scales },

        # Various supported configurations of the magnitudes feature. Will error if bins wrong?
        ** { k: _to_float_feature(v)
                    for (k, v) in mags_features.items() if k in stored_mags_features },

        # Extra features, will fall back on default value where not given
        **{ k: _to_float_feature(extra_features.get(k, d))
                    for k, d in extra_features_and_defaults.items()},
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


#
#   Helpers based on https://www.tensorflow.org/tutorials/load_data/tfrecord
#
def _to_bytes_feature(value) -> tf.train.Feature:
    """
    Encodes the passed value to a BytesList in a TensorFlow feature.

    :value: the value to encode
    :returns: the resulting Feature
    """
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    elif isinstance(value, str):
        value = bytes(value, encoding="utf8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _to_float_feature(value) -> tf.train.Feature:
    """
    Encodes the passed value to a FloatList in a TensorFlow feature.

    :value: the value to encode
    :returns: the resulting Feature
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, list):
        pass
    else:
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
