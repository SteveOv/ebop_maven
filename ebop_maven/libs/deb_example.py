
from typing import Dict, List

import numpy as np
import tensorflow as tf

label_scales = [1., 1., 1., 1., 1., 0.01, 1., 1.]
label_names = ["rA_plus_rB", "k", "J", "ecosw", "esinw", "inc", "bP", "L3"]
label_predict_cols = label_names + [f"{f}_sigma" for f in label_names]
lc_len = 1024

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

    # Complex features: phase folded light-curve
    "lc": tf.io.FixedLenFeature((lc_len,), tf.float32, [0.] * lc_len),

    # Extra features are expected to have a single float value each
    **{ k: tf.io.FixedLenFeature([], tf.float32, default_value=v)
                        for k, v in extra_features_and_defaults.items() },
}

def serialize(identifier: str,
              labels: Dict[str, float],
              light_curve_model: List[float],
              extra_features: Dict[str, float]) -> bytearray:
    """
    Encode the requested test instance data to a TF Example protobuf.

    :identifier: unique id of the instance
    :labels: the training labels
    :light_curve_model: the phase folded time series magnitude data
    :extra_features: single value features associated with the light-curve
    :returns: a bytearray containing the serialized data
    """
    if len(light_curve_model) != description["lc"].shape[0]:
        raise ValueError("Expect light_curve_model to have length " \
                        + f"{description['lc'].shape[0]}")

    features = {
        "id": _bytes_feature(identifier),
        # Labels - will raise KeyError if an expected label is not given
        **{ k: _float_feature(labels[k]) for k in label_names },
        # Light-curve model feature
        "lc" : _float_feature(light_curve_model),
        # Extra features, will fall back on default value where not given
        **{ k: _float_feature(extra_features.get(k, d))
                    for k, d in extra_features_and_defaults.items()},
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def map_parse_deb_example(record_bytes):
    """ 
    Used by TFRecordDataset().map() to deserialize the passed raw tfrecord
    row into the phase folded light-curve feature and accompanying labels
    (via a deb TF Example protobuf).

    :record_bytes: the raw byte string data of the tfrecord row
    :returns: ((lc_feature, extra_features), labels) as
    ((shape[#datapoints, 1], shape[#features, 1]), shape[#labels])
    """
    example = tf.io.parse_single_example(record_bytes, description)

    # Need to adjust the LC slightly to get it into a shape for the model
    # so basically a change from (#datapoints,) to (#datapoints, 1)
    lc_feature = tf.reshape(example["lc"], shape=(description["lc"].shape[0], 1))

    # The Extra features: ignore unknown fields and use default if not found
    ext_features = [example.get(k, d) for k, d in extra_features_and_defaults.items()]
    ext_features = tf.reshape(ext_features, shape=(len(ext_features), 1))

    # Copy labels in the expected order & apply any scaling
    labels = [example[k] * s for k, s in zip(label_names, label_scales)]
    return ((lc_feature, ext_features), labels)


#
#   Helpers based on https://www.tensorflow.org/tutorials/load_data/tfrecord
#
def _bytes_feature(value) -> tf.train.Feature:
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

def _float_feature(value) -> tf.train.Feature:
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
