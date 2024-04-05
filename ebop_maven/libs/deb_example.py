"""
Functions for reading and writing to deb_example encoded TensorFlow datasets
"""
from typing import Dict, List, Callable

import numpy as np
import tensorflow as tf

label_scales = [1., 1., 1., 1., 1., 0.01, 1., 1.]
label_names = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc", "bP", "L3"]
label_predict_cols = label_names + [f"{f}_sigma" for f in label_names]

# We can store multiple configuration of the mags feature but we can publish only one.
stored_mags_features = {
    "mags_1024_0.75": (1024, 0.75),
    "mags_1024_1.0": (1024, 1.0),
    "mags_2048_0.75": (2048, 0.75),
    "mags_2048_1.0": (2048, 1.0)
}
pub_mags_key = "mags_1024_0.75"                         # pylint: disable=invalid-name
(mags_bins, _) = stored_mags_features[pub_mags_key]     # pylint: disable=invalid-name

# Python 3.7+ language spec ensures dictionary order is preserved
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
                        for k in label_names },

    # Complex features: multiple configs of the phase folded light-curve mags
    **{ k: tf.io.FixedLenFeature((bins,), tf.float32, [0.] * bins)
                        for (k, (bins, _)) in stored_mags_features.items() },

    # Extra features are expected to have a single float value each
    **{ k: tf.io.FixedLenFeature([], tf.float32, default_value=v)
                        for k, v in extra_features_and_defaults.items() },
}

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
    if pub_mags_key not in mags_features:
        raise ValueError(f"The published {pub_mags_key} item found in mags_features.")

    features = {
        "id": _to_bytes_feature(identifier),

        # Labels - will raise KeyError if an expected label is not given
        **{ k: _to_float_feature(labels[k]) for k in label_names },

        # Various supported configurations of the magnitudes feature. Will error if bins wrong?
        ** { k: _to_float_feature(v)
                    for (k, v) in mags_features.items() if k in stored_mags_features },

        # Extra features, will fall back on default value where not given
        **{ k: _to_float_feature(extra_features.get(k, d))
                    for k, d in extra_features_and_defaults.items()},
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def create_map_func(noise_stddev: Callable[[], float] = None,
                    roll_steps: Callable[[], int] = None) -> Callable:
    """
    Configures and returns a dataset map function for deb_examples. The map function
    is used by TFRecordDataset().map() to deserialize each raw tfrecord row into the
    corresponding features and labels required for model training or testing.
    
    In addition to deserializing the rows, the map function supports the option of
    perturbing the output magnitudes feature by adding Gaussian noise and/or rolling
    the bins left or right. These options are configured by supplying functions which
    set their stddev or roll values respectively (otherwise they do nothing). Note,
    these will be running within the context of a graph so you should use tf.functions
    if you want to do anything beyond setting a fixed value. For example:

    roll_steps = lambda: tf.random.uniform([], -3, 4, tf.int32)

    :noise_stddev: a function which returns the stddev of the Gaussian noise to add
    :roll_steps: a function which returns the number of steps to roll the mag data,
    negative values roll to the left and positive values to the right
    :returns: the configured map function
    """
    # Define the map function with the two, optional perturbing actions on the mags feature
    def map_func(record_bytes):
        example = tf.io.parse_single_example(record_bytes, description)

        # Need to adjust the model mags slightly to get it into a shape for the model
        # so basically a change from (#bins,) to (#bins, 1)
        mags_feature = tf.reshape(example[pub_mags_key], shape=(mags_bins, 1))

        # Apply any perturbations to the model mags
        if noise_stddev:
            stddev = noise_stddev()
            if stddev != 0.:
                mags_feature += tf.random.normal(mags_feature.shape, stddev=stddev)

        if roll_steps:
            roll_by = roll_steps()
            if roll_by != 0:
                mags_feature = tf.roll(mags_feature, [roll_by], axis=[0])

        # The Extra features: ignore unknown fields and use default if not found
        ext_features = [example.get(k, d) for k, d in extra_features_and_defaults.items()]
        ext_features = tf.reshape(ext_features, shape=(len(ext_features), 1))

        # Copy labels in the expected order & apply any scaling
        labels = [example[k] * s for k, s in zip(label_names, label_scales)]
        return ((mags_feature, ext_features), labels)
    return map_func


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
