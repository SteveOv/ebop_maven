""" Estimator classes from which model specific ones are derived. """
from typing import List, Dict, Union
from datetime import datetime
from pathlib import Path
from abc import ABC
from inspect import getsourcefile

import numpy as np
import tensorflow as tf
from keras import Model

from . import modelling, deb_example

class Estimator(ABC):
    """ An estimator for the CNN model """

    def __init__(self, model: Union[Model, Path]=None, iterations: int=1):
        """
        Initialize a new instance of this class.

        :model: a model file or a model - if None will attempt to load default
        :iterations: set to >1 to control MC Dropout iterations per prediction
        """
        if model is None:
            model = Path(getsourcefile(lambda:0)).parent / "data/estimator/default-model.keras"
        if iterations is None:
            raise TypeError("iterations must be a positive integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self._iterations = iterations

        if isinstance(model, Path):
            if not model.exists():
                raise ValueError(f"Model file '{model}' not found.")
            print(f"{self.__class__.__name__} loading model file '{model}'...")
            self._model = modelling.load_model(model)
            modified = datetime.fromtimestamp(model.stat().st_mtime).isoformat()
            print(f"Loaded model '{self._model.name}' last modified at {modified}.")
        elif isinstance(model, Model):
            self._model = model
            print(f"Assigned model '{self._model.name}'")
        else:
            raise TypeError("Expected model to be a Path or a tf.keras.models.Model.")

        # The output layer should hold model metadata with details of inputs and outputs
        output_layer = self._model.get_layer("Output")
        if output_layer and isinstance(output_layer, modelling.OutputLayer):
            self._metadata = output_layer.metadata
        else:
            self._metadata = {}

        # Not published
        self._extra_features_and_defaults = self._metadata.pop("extra_features_and_defaults",
                                                    deb_example.extra_features_and_defaults)
        self._labels_and_scales = self._metadata.pop("labels_and_scales",
                                                    deb_example.labels_and_scales)

        # Now set up the names for the inputs and predictions (these include 1-sigma values)
        print("Each input dict to contain:   ", ", ".join(self.input_feature_names),
              f"(with mags to have {self.mags_feature_bins} bins with the",
              f"phases after {self.mags_feature_wrap_phase} wrapped)")
        print("Each output dict will publish:", ", ".join(self.label_names),
              "(and corresponding <key>_sigma uncertainty values)")

    @property
    def name(self) -> str:
        """ Gets the name of the underlying model. """
        return self._model.name

    @property
    def mags_feature_bins(self) -> int:
        """ The expected number of bins in each phase folded input mags feature """
        return self._model.input_shape[0][1]

    @property
    def mags_feature_wrap_phase(self) -> float:
        """ The expected phase after which the mags feature is wrapped. """
        return self._metadata.get("mags_wrap_phase", deb_example.default_mags_wrap_phase)

    @property
    def input_feature_names(self) -> List[str]:
        """ The names to give the input features """
        return ["mags"] + [*self._extra_features_and_defaults.keys()]

    @property
    def label_names(self) -> List[str]:
        """ Gets the ordered list of the names of the labels used to train the model. """
        return [*self._labels_and_scales.keys()]

    @property
    def prediction_names(self) -> List[str]:
        """ Gets the ordered list of the names of the predicted values, including error bars. """
        return [*self._labels_and_scales.keys()] \
                + [f"{k}_sigma" for k in self._labels_and_scales.keys()]

    @property
    def metadata(self) -> Dict[str, any]:
        """ Publish the metadata stored in the underlying model. """
        return self._metadata

    @property
    def iterations(self) -> int:
        """
        The number of iterations for each prediction.
        The MC Dropout algorithm will be used if this is greater than 1.
        """
        return self._iterations

    def predict(self,
                instances: List[Dict[str, any]],
                iterations: int=None,
                unscale: bool=True,
                seed: int=42) -> List[Dict[str, float]]:
        """
        High level call to make predictions on one or more instances' features.
        The instances' features are expected in the form of a List of dicts:

        [
          { "mags": List[float] * mags_bins, "ext_feature_0": float, "ext_feature_1": float },
        ]
        
        with the results being in the form of a List of dicts:
        
        [
          { "rA_plus_rB": float, "k": float,..., "rA_plus_rB_sigma": float, "k_sigma": float,... },
        ]

        The equivalent predict_raw() call performs the same predictions without
        the convenience of the dictionaries, instead taking and returning ndarrays.

        :instances: list of dicts, one for each instance to predict
        :iterations: the number of MC Dropout iterations (overriding the instance default)
        :unscale: indicates whether to undo the scaling of the predicted values. For example,
        the model may predict inc*0.01 and unscale would undo this returning inc as prediction/0.01
        :seed: random seed for Tensorflow
        :returns: a list dictionaries, each row the predicted labels for the matching input instance
        """
        iterations = self._iterations if iterations is None else iterations
        is_mc = iterations > 1

        # It's possible we can be given the instances as a List[Dict] but handle being given an
        # ndarray[Dict] or a single instance as a Dict: we'll process these all as ndarrays.
        if isinstance(instances, np.ndarray):
            pass
        elif isinstance(instances, List): # Convert list to ndarray
            instances = np.array(instances)
        elif isinstance(instances, Dict): # Single instance
            instances = np.array([instances])
        else:
            raise TypeError(f"Expected instances as a list of dicts but got {type(instances)}.")

        # Extract the mags_features into an ndarray of shape(#insts, #bins), as expected by
        # predict_raw(). Will throw a KeyError if the mandatory mags item is missing in any instance
        mags_features = np.array([inst["mags"] for inst in instances])

        # We need the extra_features values in an ndarray of shape (#insts, #extra_features)
        # in the correct order. Read the expected features from the input dicts, falling back
        # on the default value if not found, and ignore any unexpected key/values.
        efd = self._extra_features_and_defaults
        extra_values = [[inst_dict.get(k, df) for k, df in efd.items()] for inst_dict in instances]
        extra_values = np.array(extra_values)

        # This returns the stacked predictions in shape (#insts, #labels, #iterations)
        stkd_prds = self.predict_raw(mags_features, extra_values, iterations, unscale, seed)

        # Summarize the label predictions & append 1-sigma values to each inst
        # We go from shape (#insts, #labels, #iters) to shape (#insts, #labels*2)
        if is_mc:
            # preds are statistical mean and stddev over the iterations axis
            psets = np.concatenate([np.mean(stkd_prds, axis=2), np.std(stkd_prds, axis=2)], axis=1)
        else:
            # preds are the given values, with 1-sigmas of zero. We know the iters axis is size 1.
            nominals = stkd_prds[..., 0]
            psets = np.concatenate([nominals, np.zeros_like(nominals)], axis=1)

        # Load the predictions back into a list of dicts
        return [dict(zip(self.prediction_names, pset)) for pset in psets]


    def predict_raw(self,
                    mags_feature: np.ndarray,
                    extra_features: np.ndarray,
                    iterations: int=None,
                    unscale: bool=True,
                    seed: int=42) -> np.ndarray:
        """
        Make predictions on one or more instances' features. The instances are
        in the form of two NDArrays, one for the instances' mags data in the shape
        (#insts, #mags_bins, 1) or (#insts, #mags_bins) and another for
        extra features in the shape (#insts, #features, 1) or (#insts, #features).
        The predictions are returned as an NDArray of shape (#insts, #labels, #iterations).
        
        :instances: list of dicts, one for each instance to predict
        :mags_feature: numpy NDArray of shape (#insts, #bins, 1)
        :extra_features: numpy NDArray of shape (#insts, #features, 1)
        :iterations: the number of MC Dropout iterations (overriding the instance default)
        :unscale: indicates whether to undo the scaling of the predicted values. For example,
        the model may predict inc*0.01 and unscale would undo this returning inc as prediction/0.01
        :seed: random seed for Tensorflow
        :returns: a numpy NDArray with the predictions in the shape (#insts, #labels, #iterations)
        """
        # pylint: disable=too-many-arguments
        iterations = self._iterations if iterations is None else iterations
        is_mc = iterations > 1

        if not isinstance(mags_feature, np.ndarray):
            raise TypeError("Expect mags_feature to be an numpy ndarray")
        if not isinstance(extra_features, np.ndarray):
            raise TypeError("Expect extra_features to be a numpy ndarray")

        # Check that number of mags and feature rows (#insts) match up
        insts = mags_feature.shape[0]
        if insts != extra_features.shape[0]:
            raise ValueError("Mismatched number of mags_features and extra_features: " +
                            f"{insts} != {extra_features.shape[0]}")

        # Check the arrays are in the expected shape; (#insts, #bins, 1) and (#insts, #features, 1)
        if len(mags_feature.shape) > 1 and mags_feature.shape[1] == self.mags_feature_bins:
            if len(mags_feature.shape) == 2:
                mags_feature = mags_feature[:, :, np.newaxis]
        else:
            raise ValueError(f"Expected mags features of shape ({insts}, {self.mags_feature_bins})"+
                        f" or ({insts}, {self.mags_feature_bins}, 1) but got {mags_feature.shape}")

        count_ext_features = len(self.input_feature_names)-1
        if len(extra_features.shape) > 1 and extra_features.shape[1] == count_ext_features:
            if len(extra_features.shape) == 2:
                extra_features = extra_features[:, :, np.newaxis]
        else:
            raise ValueError(f"Expected extra features of shape ({insts}, {count_ext_features}) " +
                            f"or ({insts}, {count_ext_features}, 1) but got {extra_features.shape}")

        # If dropout, we make multiple predictions for each inst with training switched on so that
        # each prediction is with a statistically unique subset of the model's net: the MC Dropout
        # algorithm. Predictions are output in shape (#iterations, #insts, #labels)
        tf.random.set_seed(seed)
        stkd_prds = np.stack([
            self._model((mags_feature, extra_features), training=is_mc)
            for _ in range(iterations)
        ])

        # Undo any scaling applied to the labels (e.g. the model predicts inc/100)
        if unscale:
            stkd_prds /= [*self._labels_and_scales.values()]

        # Return the predictions in the shape (#insts, #labels, #iterations)
        return stkd_prds.transpose([1, 2, 0])
