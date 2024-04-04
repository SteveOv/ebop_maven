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
    _attrs = {}
    _ATTR_CREATE_TS = "created_timestamp"

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

            create_ts = self._attrs.get(self._ATTR_CREATE_TS, None) or None
            if create_ts:
                print(f"Loaded '{self._model.name}' which was created at {create_ts}.")
            else:
                modified = datetime.fromtimestamp(model.stat().st_mtime).isoformat()
                print(f"Loaded '{self._model.name}' last modified at {modified}.")
        elif isinstance(model, Model):
            self._model = model
            print(f"Assigned '{self._model.name}'")
        else:
            raise TypeError("Expected model to be a Path or a tf.keras.models.Model.")

        print(f"\tInputs: {self._model.input_shape}")
        print(f"\tOutputs: {self._model.output_shape}")

    @property
    def name(self) -> str:
        """ Gets the name of the underlying model. """
        return self._model.name

    @property
    def lc_feature_bins(self) -> int:
        """ The expected number of bins in each phase folded input mags feature """
        return self._model.input_shape[0][1]

    @property
    def iterations(self) -> int:
        """
        The number of iterations for each prediction.
        The MC Dropout algorithm will be used if this is greater than 1.
        """
        return self._iterations

    @property
    def attrs(self) -> Dict:
        """ The dictionary of attributes saved with the current model """
        return self._attrs

    def predict(self,
                instances: List[Dict[str, any]]) -> List[Dict[str, float]]:
        """
        Make predictions on one or more instances' features. The instances are
        in the form of a List of dicts.

        [
            { "lc": List[float] * mags_bins, "phiS": float, "dS_over_dP": float },
        ]
        
        :instances: list of dicts, one for each instance to predict
        :returns: a list dictionaries, each row the predicted labels for the matching input instance
        """
        is_mc = self._iterations > 1

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

        # Extract the lc_features into an ndarray of shape(#insts, #bins, 1) as expected by the
        # ML model. This will throw a KeyError if the mandatory lc item is missing in any instance.
        inst_count = instances.shape[0]
        lc_features = np.array([inst["lc"] for inst in instances])
        if lc_features.shape == (inst_count, self.lc_feature_bins, 1):
            pass
        elif lc_features.shape == (inst_count, self.lc_feature_bins, ):
            lc_features = lc_features[:, :, np.newaxis]
        else:
            raise ValueError("Expected the list of lc features to be of shape " +
                            f"(#insts, {self.lc_feature_bins}, 1) but got {lc_features.shape}")

        # We need the extra_features values in an ndarray of shape (#insts, #extra_features, 1)
        # in the correct order. Read the expected features from the input dicts, falling back
        # on the default value if not found, and ignore any unexpected key/values.
        efd = deb_example.extra_features_and_defaults
        extra_values = [[inst_dict.get(k, df) for k, df in efd.items()] for inst_dict in instances]
        extra_values = np.array(extra_values).reshape((inst_count, len(efd), 1))

        # Now check that number of lc and feature rows (#insts) match up
        if inst_count != extra_values.shape[0]:
            raise ValueError("Mismatched number of mags_features and extra_features: " +
                            f"{inst_count} != {extra_values.shape[0]}")

        # If dropout, we make multiple predictions for each inst with training switched on so that
        # each prediction is with a statistically unique subset of the model's net: the MC Dropout
        # algorithm. Stacked predictions are output in shape (#iterations, #insts, #labels)
        stkd_prds = np.stack([
            self._model((lc_features, extra_values), training=is_mc)
            for _ in range(self._iterations)
        ])

        # Undo any scaling applied to the labels (e.g. the model predicts inc/100)
        stkd_prds /= deb_example.label_scales

        # Summarize the label predictions & append 1-sigma values to each inst
        # We go from shape (#iters, #insts, #labels) to shape (#insts, #labels*2)
        if is_mc: # preds are statistical mean and stddev over the iters axis
            psets = np.concatenate([np.mean(stkd_prds, axis=0), np.std(stkd_prds, axis=0)], axis=1)
        else: # preds are the given values, with 1-sigmas of zero. Just drop the iters axis.
            nominals = stkd_prds[0, ...]
            psets = np.concatenate([nominals, np.zeros_like(nominals)], axis=1)

        # Load the predictions back into a list of dicts
        return [dict(zip(deb_example.label_predict_cols, pset)) for pset in psets]
