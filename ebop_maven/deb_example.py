"""
Details of the deb_example protobuf and support functions
"""
from typing import Dict, List, Tuple, Union

import numpy as np
from uncertainties import unumpy
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

    # These are not expected to be used as features but may be useful for filtering
    "depthP": 0.,
    "depthS": 0.,
    "excluded": 0.,
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

def create_mags_feature(phases: np.ndarray[float],
                        delta_mags: np.ndarray[float],
                        delta_mags_err: np.ndarray[float]=None,
                        num_bins: int=default_mags_bins,
                        phase_pivot: float=None,
                        include_phases: bool=False) \
                           -> Union[np.ndarray[float], Tuple[np.ndarray[float], np.ndarray[float]]]:
    """
    Create a binned copy of the passed light curve delta mags data for use as a mags feature.

    By default this just returns an array of the binned mags. However if include_phases is True
    this returns a tuple of arrays, one each for the binned phases and binned mags.

    :phases: the normalized phases of the delta_mags data
    :delta_mags: and the related delta_mags values (in units of mag)
    :delta_mags_err: the optional uncertainties of the delta_mags data
    :num_bins: the number of equally spaced bins to populate
    :phase_pivot: the pivot point about which the fold phase was wrapped to < 0;
    inferred from the maximum phase value if omitted
    :include_phases: whether to return a tuple of (phases, mags) or just the mags
    :returns: a tuple with requested number of binned phases and delta magnitudes
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    if delta_mags_err is not None:
        delta_mags = unumpy.uarray(delta_mags, delta_mags_err)

    # If there is a phase wrap then phases above the pivot will have been
    # wrapped around to <0. Work out what the expected minimum phase should be.
    if phase_pivot is not None:
        max_phase = max(phase_pivot, phases.max())
    else:
        max_phase = phases.max()
    min_phase = min(max_phase - 1, phases.min())

    # Because we will likely know the exact max phase but the min will be infered we make sure the
    # phases end at the pivot/max_phase but start just "above" the expected min phase (logically
    # equiv to startpoint=False). Working with the searchsorted side="left" arg, which allocates
    # indices where bin_phase[i-1] < src_phase <= bin_phase[i], we map all source data to a bin.
    bin_phases = np.flip(np.linspace(max_phase, min_phase, num_bins, endpoint=False))
    phase_bin_ix = np.searchsorted(bin_phases, phases, "left")

    # Perform the "mean" binning
    bin_mags = np.empty_like(bin_phases, dtype=delta_mags.dtype)
    for bin_ix in range(num_bins):
        # np.where() indices are quicker than masking
        phase_ix = np.where(phase_bin_ix == bin_ix)[0]
        if len(phase_ix) > 0:
            bin_mags[bin_ix] = delta_mags[phase_ix].mean()
        else:
            bin_mags[bin_ix] = np.nan

    # We only need the nominal value for the mags feature
    if bin_mags.dtype == np.dtype(object): # UFloat
        bin_mags = unumpy.nominal_values(bin_mags)

    # Fill any gaps by interpolation; we have a np.nan where there were no source data within a bin
    if any(missing := np.isnan(bin_mags)):
        def equiv_ix(ix):
            return ix.nonzero()[0]
        bin_mags[missing] = np.interp(equiv_ix(missing), equiv_ix(~missing), bin_mags[~missing])

    if include_phases:
        return bin_phases, bin_mags
    return bin_mags

def serialize(identifier: str,
              labels: Dict[str, float],
              phases: np.ndarray[float],
              delta_mags: np.ndarray[float],
              delta_mags_err: np.ndarray[float]=None,
              extra_features: Dict[str, float]=None) -> bytearray:
    """
    Encode the requested test instance data to a TF Example protobuf.

    :identifier: unique id of the instance
    :labels: the training labels
    :phases: the normalized phases of the delta_mags data
    :delta_mags: and the related delta_mags values (in units of mag)
    :delta_mags_err: optional uncertainties of the delta_mags data
    :extra_features: optional, single value features associated with the light-curve
    :returns: a bytearray containing the serialized data
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    if delta_mags_err is not None:
        delta_mags = unumpy.uarray(delta_mags, delta_mags_err)
    phase_pivot = phases.max()

    if extra_features is None:
        extra_features = {}

    features = {
        "id": _to_bytes_feature(identifier),

        # Labels - will raise KeyError if an expected label is not given
        **{ k: _to_float_feature(labels[k]) for k in labels_and_scales },

        # Any supported configurations of the magnitudes feature derived from the
        # phase-folded light curve data passed in (ignore the binned phase output)
        **{ k : _to_float_feature(create_mags_feature(phases, delta_mags, None,
                                                      num_bins, phase_pivot, False))
                    for k, num_bins in stored_mags_features.items() },

        # Extra features, will fall back on default value where not given
        **{ k: _to_float_feature(extra_features.get(k, default))
                    for k, default in extra_features_and_defaults.items() },
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
