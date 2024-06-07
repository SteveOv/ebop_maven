"""
Functions for reading and writing to deb_example encoded TensorFlow datasets
"""
from typing import Dict, List, Tuple, Callable, Iterable, Union
from pathlib import Path

import numpy as np
import tensorflow as tf

# We can store multiple configuration of the mags feature but we can publish only one.
stored_mags_features = {
    "mags_1024_0.75": (1024, 0.75),
    "mags_1024_1.0": (1024, 1.0),
    "mags_2048_0.75": (2048, 0.75),
    "mags_2048_1.0": (2048, 1.0),
    "mags_4096_0.75": (4096, 0.75),
    "mags_4096_1.0": (4096, 1.0),
}
default_mags_key = "mags_1024_0.75"                                                     # pylint: disable=invalid-name
(default_mags_bins, default_mags_wrap_phase) = stored_mags_features[default_mags_key]   # pylint: disable=invalid-name


# Python 3.7+ language spec ensures dictionary order is preserved
# The full set of stored labels and any scaling applied to them when read
labels_and_scales = {
    "rA_plus_rB": 1.,
    "k": 1.,
    "J": 1.,
    "ecosw": 1.,
    "esinw": 1.,
    "inc": 0.01,
    "bP": 1.,
    "L3": 1.,
    "cosi": 1.,
    "sini": 1.
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
                        for (k, (bins, _)) in stored_mags_features.items() },

    # Extra features are expected to have a single float value each
    **{ k: tf.io.FixedLenFeature([], tf.float32, default_value=v)
                        for k, v in extra_features_and_defaults.items() },
}

# Common options used when reading or writing a deb Example dataset file
ds_options = tf.io.TFRecordOptions(compression_type=None)

def create_mags_key(mags_bins: int, mags_wrap_phase: float) -> str:
    """ Helper function to format a key to the stored_mags_features dict """
    return f"mags_{int(mags_bins)}_{float(mags_wrap_phase)}"

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


def create_map_func(mags_bins: int = default_mags_bins,
                    mags_wrap_phase: float = default_mags_wrap_phase,
                    ext_features: List[str] = None,
                    labels: List[str] = None,
                    noise_stddev: Callable[[], float] = None,
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

    :mags_bins: the width of the mags to publish
    :mags_wrap_phase: the wrap phase of the mags to publish
    :ext_features: a chosen subset of the available ext_features,
    in the requested order, or all if None
    :labels: a chosen subset of the available labels, in requested order, or all if None
    :noise_stddev: a function which returns the stddev of the Gaussian noise to add
    :roll_steps: a function which returns the number of steps to roll the mag data,
    negative values roll to the left and positive values to the right
    :returns: the configured map function
    """
    mags_key = create_mags_key(mags_bins, mags_wrap_phase)
    if ext_features is not None:
        chosen_ext_feat_and_defs = { ef: extra_features_and_defaults[ef] for ef in ext_features}
    else:
        chosen_ext_feat_and_defs = extra_features_and_defaults
    if labels is not None:
        chosen_lab_and_scl = { l: labels_and_scales[l] for l in labels }
    else:
        chosen_lab_and_scl = labels_and_scales

    # Define the map function with the two, optional perturbing actions on the mags feature
    def map_func(record_bytes):
        example = tf.io.parse_single_example(record_bytes, description)

        # Need to adjust the model mags slightly to get it into a shape for the model
        # so basically a change from (#bins,) to (#bins, 1)
        mags_feature = tf.reshape(example[mags_key], shape=(mags_bins, 1))

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
        ext_features = [example.get(k, d) for (k, d) in chosen_ext_feat_and_defs.items()]
        ext_features = tf.reshape(ext_features, shape=(len(ext_features), 1))

        # Copy labels in the expected order & apply any scaling
        labels = [example[k] * s for k, s in chosen_lab_and_scl.items()]
        return ((mags_feature, ext_features), labels)
    return map_func


def create_dataset_pipeline(dataset_files: Iterable[str],
                            batch_size: float=100,
                            map_func: Callable=create_map_func(),
                            filter_func: Callable=None,
                            shuffle: bool=False,
                            reshuffle_each_iteration: bool=False,
                            max_buffer_size: int=1000000,
                            prefetch: int=0,
                            seed: int=42) -> Tuple[tf.data.TFRecordDataset, int]:
    """
    Creates the requested TFRecordDataset pipeline.

    :dataset_files: the source tfrecord dataset files.
    :batch_size: the relative size of each batch. May be set to 0 (no batch, effectively all rows),
    <1 (this fraction of all rows) or >=1 this size (will be rounded)
    :map_func: the map function to use to deserialize each row
    :filter_func: an optional func to filter the results (must be a tf.function)
    :shuffle: whether to include a shuffle step in the pipeline
    :reshuffle_each_iteration: whether the shuffle step suffles on each epoch
    :max_buffer_size: the maximum size of the shuffle buffer
    :prefetch: the number of prefetch operations to perform
    :seed: seed for any random behaviour
    :returns: a tuple of (dataset pipeline, row count). The row count is the total
    rows without any optional filtering applied.
    """
    # Read through once to get the total number of records
    ds = tf.data.TFRecordDataset(list(dataset_files), num_parallel_reads=100,
                                 compression_type=ds_options.compression_type)
    row_count = ds.reduce(0, lambda count, _: count+1).numpy()

    # Now build the full pipeline
    if shuffle or reshuffle_each_iteration:
        buffer_size = min(row_count, max_buffer_size)
        ds = ds.shuffle(buffer_size, seed, reshuffle_each_iteration=reshuffle_each_iteration)

    ds = ds.map(map_func)

    if filter_func:
        ds = ds.filter(filter_func)

    if batch_size:
        if batch_size < 1:
            batch_size = int(np.ceil(row_count * batch_size))
        else:
            batch_size = min(row_count, int(batch_size))
        if batch_size:
            ds = ds.batch(batch_size, drop_remainder=True)

    if prefetch:
        ds = ds.prefetch(prefetch)
    return (ds, row_count)


def iterate_dataset(dataset_files: Iterable[str],
                    mags_bins: int = default_mags_bins,
                    mags_wrap_phase: float = default_mags_wrap_phase,
                    ext_features: List[str] = None,
                    labels: List[str]=None,
                    identifiers: List[str]=None,
                    scale_labels: bool=False,
                    noise_stddev: float = 0.,
                    roll_max: int = 0):
    """
    Utility/diagnostics function which will parse a saved dataset yielding rows,
    and within the rows labels and features, which match the requested criteria.
    
    The rows are yielded in the order in which they appear in the supplied dataset files.
    
    The extra_features and labels are listed in the order they have been specified or in the
    order of extra_features_and_defaults and labels_and_scales if not specified.

    This function is not for use when training a model; for that, requirement use
    create_dataset_pipeline() directly. Instead this gives easy access to the
    contents of a dataset for diagnostics lookup, testing or plotting.

    :dataset_files: the set of dataset files to parse
    :mags_bins: the width of the mags to publish
    :mags_wrap_phase: the wrap phase of the mags to publish
    :ext_features: a chosen subset of the available features, in this order, or all if None
    :labels: a chosen subset of the available labels, in this order, or all if None
    :identifiers: optional list of ids to yield, or all ids if None
    :scale_values: if True values will be scaled
    :noise_stddev: the standard deviation of Gaussian Noise to add to the mags feature
    :roll_max: the maximum random roll to apply to the mags feature
    :returns: for each matching row yields a tuple of (id, mags vals, ext feature vals, label vals)
    """
    # pylint: disable=too-many-arguments, too-many-locals
    mags_key = create_mags_key(mags_bins, mags_wrap_phase)
    if ext_features is not None:
        chosen_ext_feats = {
            f: extra_features_and_defaults[f] for f in ext_features if f != "mags" }
    else:
        chosen_ext_feats = extra_features_and_defaults
    if labels is not None:
        chosen_labels = { l: labels_and_scales[l] for l in labels }
    else:
        chosen_labels = labels_and_scales

    def map_func(record_bytes):
        example = tf.io.parse_single_example(record_bytes, description)
        mags_feature = example[mags_key]

        # Apply any required random augmentations to the model mags
        if noise_stddev:
            mags_feature += tf.random.normal(mags_feature.shape, stddev=noise_stddev)

        if roll_max:
            roll_by = tf.random.uniform([], -roll_max, roll_max+1, tf.int32)
            if roll_by != 0:
                mags_feature = tf.roll(mags_feature, [roll_by], axis=[0])

        ext_features = [example.get(k, d) for (k, d) in chosen_ext_feats.items()]
        if scale_labels:
            labels = [example[k] * s for k, s in chosen_labels.items()]
        else:
            labels = [example[k] for k in chosen_labels]
        return example["id"], mags_feature, ext_features, labels

    # Create a custom pipeline for this dataset, with the above map function
    (ds, count) = create_dataset_pipeline(dataset_files, 0, map_func)
    for id_val, mags_val, feat_vals, lab_vals in ds.take(count).as_numpy_iterator():
        id_val = id_val.decode(encoding="utf8")
        if identifiers is None or id_val in identifiers:
            yield id_val, mags_val, feat_vals, lab_vals


def read_dataset(dataset_files: Iterable[str],
                 mags_bins: int = default_mags_bins,
                 mags_wrap_phase: float = default_mags_wrap_phase,
                 ext_features: List[str] = None,
                 labels: List[str]=None,
                 identifiers: List[str]=None,
                 scale_labels: bool=False,
                 noise_stddev: float = 0.,
                 roll_max: int = 0):
    """
    Wrapper around iterate_dataset() which handles the iteration and returns separate
    np ndarrays for the dataset ids, mags values, feature values and label values.

    This may not be hugely performant, especially with large datasets, but it's for convenience.

    :dataset_files: the set of dataset files to parse
    :mags_bins: the width of the mags to publish
    :mags_wrap_phase: the wrap phase of the mags to publish
    :ext_features: a chosen subset of the available features, in this order, or all if None
    :labels: a chosen subset of the available labels, in this order, or all if None
    :identifiers: optional list of ids to yield, or all ids if None
    :scale_values: if True values will be scaled
    :noise_stddev: the standard deviation of Gaussian Noise to add to the mags feature
    :roll_max: the maximum random roll to apply to the mags feature
    :returns: Tuple[
    NDArray[#insts, 1], NDArray[#insts, #bins], NDArray[#insts, #feats], NDArray[#insts, #labels]]
    """
    # pylint: disable=too-many-arguments, too-many-locals
    ids, mags_vals, feature_vals, label_vals = [], [], [], []
    for row in iterate_dataset(dataset_files, mags_bins, mags_wrap_phase,
                               ext_features, labels, identifiers, scale_labels,
                               noise_stddev, roll_max):
        ids += [row[0]]
        mags_vals += [row[1]]
        feature_vals += [row[2]]
        label_vals += [row[3]]

    # Need to sort the data in the order of the requested ids (if given).
    # Not hugely performant, but we only ever expect short lists of indices.
    if identifiers is not None and len(identifiers) > 0:
        indices = [ids.index(i) for i in identifiers if i in ids]
        ids = [ids[ix] for ix in indices]
        mags_vals = [mags_vals[ix] for ix in indices]
        feature_vals = [feature_vals[ix] for ix in indices]
        label_vals = [label_vals[ix] for ix in indices]

    return np.array(ids), np.array(mags_vals), np.array(feature_vals), np.array(label_vals)


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
