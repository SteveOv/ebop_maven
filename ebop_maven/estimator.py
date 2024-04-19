""" Estimator classes from which model specific ones are derived. """
from typing import List, Dict, Union
from datetime import datetime
from pathlib import Path
from abc import ABC

import numpy as np
from keras.models import Model

from . import modelling
from .libs import deb_example

class Estimator(ABC):
    """ An estimator for the CNN model """

    def __init__(self, model: Union[Model, Path], iterations: int=1):
        """
        Initialize a new instance of this class.

        :model: a model file or a model - if None will attempt to load default
        :iterations: set to >1 to control MC Dropout iterations per prediction
        """
        if not model:
            raise ValueError("Model must be a valid keras Model or the Path of a saved model")
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

        # Find out about what we're predicting - it should be in the output layer
        output_layer = self._model.get_layer("Output")
        if output_layer \
                and isinstance(output_layer, modelling.OutputLayer) \
                and output_layer.label_names_and_scales:
            self._label_names_and_scales = output_layer.label_names_and_scales
        else:
            self._label_names_and_scales = deb_example.labels_and_scales

        # Now set up the names for the predictions (these include 1-sigma values)
        self._prediction_names = [*self._label_names_and_scales] \
                                + [f"{k}_sigma" for k in self._label_names_and_scales]

        print("Expects each input dict to hold:", ", ".join(self.input_feature_names))
        print(f"\tThe mags feature to be of {self.mags_feature_bins} bins length,",
              f"wrapped after phase {self.mags_feature_wrap_phase}")
        print("Each output dict will publish:  ", ", ".join(self.prediction_names))

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
        # TODO: it would be better to be able to read this from the model
        return deb_example.mags_wrap_phase

    @property
    def input_feature_names(self) -> List[str]:
        """ The names to give the input features """
        # TODO: it would be better to be able to read this from the model
        return ["mags"] + list(deb_example.extra_features_and_defaults.keys())

    @property
    def iterations(self) -> int:
        """
        The number of iterations for each prediction.
        The MC Dropout algorithm will be used if this is greater than 1.
        """
        return self._iterations

    @property
    def label_names_and_scales(self) -> Dict[str, float]:
        """ Names and scaling applied to the labels to be predicted """
        return self._label_names_and_scales

    @property
    def prediction_names(self) -> List[str]:
        """ Gets the ordered list of the names of the predicted values. """
        return self._prediction_names

    def predict(self,
                instances: List[Dict[str, any]],
                iterations: int=None,
                unscale: bool=True) -> List[Dict[str, float]]:
        """
        Make predictions on one or more instances' features. The instances are
        in the form of a List of dicts.

        [
            { "mags": List[float] * mags_bins, "phiS": float, "dS_over_dP": float },
        ]
        
        :instances: list of dicts, one for each instance to predict
        :iterations: the number of MC Dropout iterations (overriding the instance default)
        :unscale: indicates whether to undo the scaling of the predicted values. For example,
        the model may predict inc*0.01 and unscale would undo this returning inc as prediction/0.01
        :returns: a list dictionaries, each row the predicted labels for the matching input instance
        """
        iterations = self._iterations if iterations is None else iterations
        is_mc = iterations > 1

        # It's possible we can be given the instances as a List[Dict] but handle being given an
        # ndarray or a single instance as a Dict: we'll process these all as ndarrays.
        if isinstance(instances, np.ndarray):
            pass
        elif isinstance(instances, List): # Convert list to ndarray
            instances = np.array(instances)
        elif isinstance(instances, Dict): # Single instance
            instances = np.array([instances])
        else:
            raise TypeError(f"Expected instances as a list of dicts but got {type(instances)}.")

        # Extract the mags_features into an ndarray of shape(#insts, #bins, 1) as expected by the
        # ML model. This will throw a KeyError if the mandatory mags item is missing in any instance
        inst_count = instances.shape[0]
        mags_features = np.array([inst["mags"] for inst in instances])
        if mags_features.shape == (inst_count, self.mags_feature_bins, 1):
            pass
        elif mags_features.shape == (inst_count, self.mags_feature_bins, ):
            mags_features = mags_features[:, :, np.newaxis]
        else:
            raise ValueError("Expected the list of mags features to be of shape " +
                            f"(#insts, {self.mags_feature_bins}, 1) but got {mags_features.shape}")

        # We need the extra_features values in an ndarray of shape (#insts, #extra_features, 1)
        # in the correct order. Read the expected features from the input dicts, falling back
        # on the default value if not found, and ignore any unexpected key/values.
        efd = deb_example.extra_features_and_defaults
        extra_values = [[inst_dict.get(k, df) for k, df in efd.items()] for inst_dict in instances]
        extra_values = np.array(extra_values).reshape((inst_count, len(efd), 1))

        # Now check that number of mags and feature rows (#insts) match up
        if inst_count != extra_values.shape[0]:
            raise ValueError("Mismatched number of mags_features and extra_features: " +
                            f"{inst_count} != {extra_values.shape[0]}")

        # If dropout, we make multiple predictions for each inst with training switched on so that
        # each prediction is with a statistically unique subset of the model's net: the MC Dropout
        # algorithm. Stacked predictions are output in shape (#iterations, #insts, #labels)
        stkd_prds = np.stack([
            self._model((mags_features, extra_values), training=is_mc)
            for _ in range(iterations)
        ])

        # Undo any scaling applied to the labels (e.g. the model predicts inc/100)
        if unscale:
            stkd_prds /= [*self.label_names_and_scales.values()]

        # Summarize the label predictions & append 1-sigma values to each inst
        # We go from shape (#iters, #insts, #labels) to shape (#insts, #labels*2)
        if is_mc: # preds are statistical mean and stddev over the iters axis
            psets = np.concatenate([np.mean(stkd_prds, axis=0), np.std(stkd_prds, axis=0)], axis=1)
        else: # preds are the given values, with 1-sigmas of zero. Just drop the iters axis.
            nominals = stkd_prds[0, ...]
            psets = np.concatenate([nominals, np.zeros_like(nominals)], axis=1)

        # Load the predictions back into a list of dicts
        return [dict(zip(self.prediction_names, pset)) for pset in psets]
