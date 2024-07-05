""" Estimator classes from which model specific ones are derived. """
from typing import List, Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
from abc import ABC
from inspect import getsourcefile

import numpy as np
from uncertainties import ufloat, UFloat
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
        self._labels_and_scales = self._metadata.pop("labels_and_scales",\
                                                    deb_example.labels_and_scales)
        self._dtypes = [(name, UFloat) for name in self._labels_and_scales]

        print("The prediction inputs are:\n",
              f"\tmags_feature - numpy ndarray[float] shape (#instances, {self.mags_feature_bins})",
              f"with the phases after {self.mags_feature_wrap_phase} wrapped by -1")
        if len(self.extra_feature_names):
            print( "\textra_features - numpy ndarray[float] shape (#instances, ",
                  f"{len(self.extra_feature_names)}) for the features;",
                   ", ".join(self.extra_feature_names))
        else:
            print("\textra_features - numpy ndarray shape (#instances, 0) or None",
                  "as extra_features are not used for predictions")           
        print("The prediction results are:\n",
               "\tpredicted values as an numpy recarray[UFloat] of shape",
              f"(#instances, [{', '.join(self.label_names)}])\n",
               "\toptionally raw predictions as a numpy NDArray[float] of shape",
              f"(#instances, {len(self.label_names)}, #iterations), if include_raw==True")

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
    def extra_feature_names(self) -> List[str]:
        """ The names to give the input features """
        return [*self._extra_features_and_defaults.keys()]

    @property
    def label_names(self) -> List[str]:
        """ Gets the ordered list of the names of the labels used to train the model. """
        return [*self._labels_and_scales.keys()]

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
                mags_feature: np.ndarray[float],
                extra_features: np.ndarray[float]=None,
                iterations: int=None,
                unscale: bool=True,
                include_raw_preds: bool=False,
                seed: int=42) \
            -> Union[np.rec.recarray[UFloat], Tuple[np.rec.recarray[UFloat], np.ndarray[float]]]:
        """
        Make predictions on one or more instances' features. The instances are
        in the form of two NDArrays, one for the instances' mags data in the shape
        (#insts, #mags_bins, 1) or (#insts, #mags_bins) and another for
        extra features in the shape (#insts, #features, 1) or (#insts, #features).
        The predictions are returned as an recarray of UFloats of shape (#insts, #labels), with
        the labels accessible by the names given in self.label_names, and if include_raw_preds is
        set, an NDArray of the raw predictions in the shape (#insts, #labels, #iterations)

        Examples:
        ```Python
        preds = estimator.predict2(mags, ext, 1000)
        print(preds[0]['k'].nominal_value)

        # masking support
        transiting = preds[transit_flags]
        ```
        
        :mags_feature: numpy NDArray of shape (#insts, #bins, 1)
        :extra_features: numpy NDArray of shape (#insts, #features, 1)
        :iterations: the number of MC Dropout iterations (overriding the instance default)
        :unscale: indicates whether to undo the scaling of the predicted values. For example,
        the model may predict inc*0.01 and unscale would undo this returning inc as prediction/0.01
        :include_raw_preds: if True the raw preds will also be returned as the 2nd item of a tuple
        :seed: random seed for Tensorflow
        :returns: a numpy recarray[UFloat] with the predictions in the shape (#insts, #labels) and
        optionally an NDArray of the raw predictions in the shape (#insts, #labels, #iterations)
        """
        # pylint: disable=too-many-arguments, too-many-branches
        iterations = self._iterations if iterations is None else iterations
        is_mc = iterations > 1

        if not isinstance(mags_feature, np.ndarray):
            raise TypeError("Expect mags_feature to be an numpy ndarray")
        if not isinstance(extra_features, (None|np.ndarray)):
            raise TypeError("Expect extra_features to be a numpy ndarray")

        # Check the mags features are in the expected shape; (#insts, #bins) or (#insts, #bins, 1)
        insts = mags_feature.shape[0]
        if len(mags_feature.shape) > 1 and mags_feature.shape[1] == self.mags_feature_bins:
            if len(mags_feature.shape) == 2:
                mags_feature = mags_feature[:, :, np.newaxis]
        else:
            raise ValueError(f"Expected mags features of shape (#insts, {self.mags_feature_bins})"+
                        f" or (#insts, {self.mags_feature_bins}, 1) but got {mags_feature.shape}")

        # For extra features, we allow it to be None if there are no expected features
        # otherwise we expect it to be in the shape (#insts, #features) or (#insts, #features, 1)
        num_ext_features = len(self.extra_feature_names)
        if num_ext_features or extra_features is not None:
            if len(extra_features.shape) > 1 and extra_features.shape[0] == insts and \
                                                extra_features.shape[1] == num_ext_features:
                if len(extra_features.shape) == 2:
                    extra_features = extra_features[:, :, np.newaxis]
            else:
                raise ValueError(f"Expected extra features of shape ({insts}, {num_ext_features}) "+
                            f"or ({insts}, {num_ext_features}, 1) but got {extra_features.shape}")
        else:
            # No extra features required, and None supplied. Make sure we don't choke the model.
            extra_features = np.empty(shape=(insts, 0, 1))

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

        # Calc the means/stddevs, over the #iters axis to go to lists of records [(ufloats, ...)]
        if is_mc:
            # We have multiple predictions per input. Summarize over #iters axis
            records = [
                tuple(ufloat(n, s) for (n, s) in zip(noms, sigs))
                    for (noms, sigs) in zip(np.mean(stkd_prds, axis=0), np.std(stkd_prds, axis=0))
            ]
        else:
            # Not MC; only 1 set of predictions per instance so drop #iters axis and assume zero std
            records = [tuple(ufloat(n, 0) for n in noms) for noms in stkd_prds[0, ...]]

        if not include_raw_preds:
            return np.rec.fromrecords(records, dtype=self._dtypes)
        return np.rec.fromrecords(records, dtype=self._dtypes), stkd_prds.transpose([1, 2, 0])
