""" Estimator classes from which model specific ones are derived. """
from typing import List, Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
from abc import ABC
from inspect import getsourcefile
import os
import errno

import numpy as np
from uncertainties import UFloat, unumpy
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

        file_timestamp = datetime.now()
        if isinstance(model, Path):
            if not model.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model)
            print(f"{self.__class__.__name__} loading '{model}'...", end="")
            self._model = modelling.load_model(model)
            print(f"loaded '{self._model.name}'.")
            file_timestamp = datetime.fromtimestamp(model.stat().st_mtime).isoformat()
        elif isinstance(model, Model):
            self._model = model
            print(f"{self.__class__.__name__} assigned model '{self._model.name}'")
        else:
            raise TypeError("Expected model to be a Path or a tf.keras.models.Model.")

        # The output layer should hold model metadata with details of inputs and outputs
        output_layer = self._model.get_layer("Output")
        if output_layer \
                and isinstance(output_layer, (modelling.OutputLayer, modelling.OutputLayerConcat)):
            self._metadata = output_layer.metadata
        else:
            self._metadata = {}
        self._metadata.setdefault("created_timestamp", file_timestamp)

        # Not published
        self._extra_features_and_defaults = self._metadata.pop("extra_features_and_defaults",
                                                    deb_example.extra_features_and_defaults)
        self._labels_and_scales = self._metadata.pop("labels_and_scales",\
                                                    deb_example.labels_and_scales)
        self._dtypes = [(name, np.dtype(UFloat.dtype)) for name in self._labels_and_scales]
        self._scale_values = list(self._labels_and_scales.values())
        self._scaling_applied = any(s != 1 for s in self._scale_values)

        print(f"The model {self.name} was created at {self._metadata['created_timestamp']}")

        print("The input features are:")
        print(f"  mags_feature as a NDarray[float] of shape (#instances, {self.mags_feature_bins})",
              "containing a phase folded light curve ", end="")
        if self.mags_feature_wrap_phase is None:
            print("centred on the mid-point between the primary and secondary eclipse")
        elif self.mags_feature_wrap_phase:
            print(f"with the phases above {self.mags_feature_wrap_phase} wrapped by -1")
        if len(self.extra_feature_names):
            print("  extra_features as a NDarray[float] of shape (#instances,",
                  f"{len(self.extra_feature_names)}) for [{', '.join(self.extra_feature_names)}])")
        else:
            print( "  extra_features as a NDarray of shape (#instances, 0) or None,",
                  "as no extra_features are used for predictions")

        print("The prediction results are:")
        print( "  predicted values as a structured NDarray[UFloat] of shape",
              f"(#instances, [{', '.join(self.label_names)}])")
        print("  optionally, if include_raw is True, all MC predictions as a NDArray[float]",
              f"of shape (#instances, {len(self.label_names)}, #iterations)")

    def __str__(self):
        """ Uses the model name as a string representation of the class. """
        return self._model.name

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
                include_raw_preds: bool=False) \
            -> Union[np.ndarray[UFloat], Tuple[np.ndarray[UFloat], np.ndarray[float]]]:
        """
        Make predictions on one or more instances' features. The instances are
        in the form of two numpy NDArrays, mags_feature and extra_features:
        - the mags_feature must be in a form from which we can infer the shape (#insts, #bins, 1)
            - for example: (#insts, #bins); or (#bins, 1) or (#bins, ) only if there is 1 inst
        - the ext_features must be in a form from which we can infer the shape (#insts, #feats, 1)
            - for example: (#insts, #feats); or (#feats, 1) or (#feats, ) only if ther is 1 inst
            - #insts, inferred or actual, must match the equivalent value for the mags data
            - set to None if extra features are not required

        Prior to making any predictions a check for nans will be made on the mags_feature bins. If
        any are found, substitute values will be derived by simple 1-D linear interpolation. Those
        wanting a custom nan strategy should implment it in client code before calling predict().

        The predictions are returned as a structured NDArray of shape (#insts, #labels), with the
        labels accessible by the names given in self.label_names. If iterations is 1, the NDArray
        will contain the predicted values as UFloats with uncertainty values of zero. If
        iterations > 1 the NDarray will contain UFloats with the nominal and uncertainty values set
        to the mean and 1-sigma of each of the predicted values over all of the iterations.
        
        If include_raw_preds is True, a Tuple is returned with the predictions and a second NDArray
        of floats with each iteration's predictions in the shape (#insts, #labels, #iterations).

        Examples:
        ```Python
        preds = estimator.predict(mags, ext_features, 1000)

        # Using named columns        
        ratio_radii = preds[0]['k'].nominal_value

        # Using masked rows
        preds_with_larger_secondary = preds[preds['k'] > 1.0]

        # Support for raw_preds
        (preds, raw_preds) = estimator.predict(mags, ext_features, 1000, include_raw_preds=True)
        ```
        
        :mags_feature: numpy NDArray in shape that can be interpreted as (#insts, #bins, 1)
        :extra_features: numpy NDArray in shape that can be interpreted as (#insts, #features, 1)
        :iterations: the number of MC Dropout iterations (overriding the instance default)
        :unscale: indicates whether to undo the scaling of the predicted values. For example,
        the model may predict inc*0.01 and unscale would undo this returning inc as prediction/0.01
        :include_raw_preds: if True the raw preds will also be returned as the 2nd item of a tuple
        :returns: a structured NDarray[UFloat] with the predictions in the shape (#insts, #labels)
        and optionally an NDArray of the raw predictions in the shape (#insts, #labels, #iterations)
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-branches
        iterations = self._iterations if iterations is None else iterations
        is_mc = iterations > 1

        if not isinstance(mags_feature, np.ndarray):
            raise TypeError("Expect mags_feature to be an numpy ndarray")
        if extra_features is not None and not isinstance(extra_features, np.ndarray):
            raise TypeError("Expect extra_features to be a numpy ndarray")

        # For the model we need the mags features in the expected shape of (#insts, #bins, 1).
        # We can infer the missing #insts & trailing dim so long as we can find the #bins axis.
        # Not that expand_dims creates a view on the source array, so leaves it unchanged.
        nbins = self.mags_feature_bins
        if mags_feature.ndim == 3 and mags_feature.shape[1] == nbins and mags_feature.shape[2] == 1:
            ninsts = mags_feature.shape[0]
        elif mags_feature.ndim == 2 and mags_feature.shape[1] == nbins:
            ninsts = mags_feature.shape[0]
            mags_feature = np.expand_dims(mags_feature, axis=2)
        elif mags_feature.shape == (nbins, 1):
            ninsts = 1
            mags_feature = np.expand_dims(mags_feature, axis=0)
        elif mags_feature.shape == (nbins, ):
            ninsts = 1
            mags_feature = np.expand_dims(mags_feature, axis=(0, 2))
        else:
            raise ValueError("Expected mags_feature in a shape which could be infered as " +
                             f"(#insts, {nbins}, 1) but got shape {mags_feature.shape}")

        # For extra features we allow it to be None if there are no expected features,
        # otherwise the model expects it in the shape (#insts, #features, 1).
        # Here infering missing dims is easier as we now know #insts from the mags feature above.
        nfeats = len(self.extra_feature_names)
        if nfeats or extra_features is not None:
            if extra_features.shape == (ninsts, nfeats, 1):
                pass
            elif extra_features.shape == (ninsts, nfeats):
                extra_features = np.expand_dims(extra_features, axis=2)
            elif ninsts == 1 and extra_features.shape == (nfeats, 1):
                extra_features = np.expand_dims(extra_features, axis=0)
            elif ninsts == 1 and extra_features.shape == (nfeats, ):
                extra_features = np.expand_dims(extra_features, axis=(0, 2))
            else:
                raise ValueError("Expected extra_features in a shape which could be infered as " +
                                 f"({ninsts}, {nfeats}, 1) but got shape {extra_features.shape}")
        else:
            # No extra features required, and None supplied. Make sure we don't choke the model.
            extra_features = np.empty(shape=(ninsts, 0, 1), dtype=float)

        # Replace any gaps (nan values) in the mags data with linearly interpolated values. If we
        # need to fill gaps, do it on a copy so as not to surprise client code by changing its data.
        if np.any(np.isnan(mags_feature)):
            mags_feature = mags_feature.copy()
            for inst in np.arange(ninsts):
                deb_example.interpolate_nan_mags(mags_feature[inst])

        # If dropout, we make multiple predictions for each inst with training switched on so that
        # each prediction is with a statistically unique subset of the model's net: the MC Dropout
        # algorithm. These raw predictions will be in the shape (#iterations, #insts, #labels)
        raw_preds = np.stack([
            self._model((mags_feature, extra_features), training=is_mc)
            for _ in range(iterations)
        ])

        if unscale and self._scaling_applied:
            # Undo any scaling applied to the labels (e.g. the model may predict inc/100).
            # The scales list matches the size of the final labels dimension of the raw preds and
            # it is broadcast over the other, iterations and instances, dimensions when re-scaling.
            raw_preds /= self._scale_values

        preds = np.empty(shape=(ninsts, ), dtype=self._dtypes) # pre-allocate the results
        if is_mc:
            # We have multiple predictions per input. Summarize over the iterations axis
            for inst, (noms, errs) in enumerate(zip(np.mean(raw_preds, axis=0),
                                                    np.std(raw_preds, axis=0))):
                preds[inst] = tuple(unumpy.uarray(noms, errs))
        else:
            # Not MC; only 1 set of predictions per instance so ignore #iters and assume zero std
            for inst in range(ninsts):
                preds[inst] = tuple(unumpy.uarray(raw_preds[0, inst, ...], 0))

        if not include_raw_preds:
            return preds
        return preds, raw_preds.transpose([1, 2, 0])
