""" Unit tests for the TensorFlowEstimator base class. """
# pylint: disable=too-many-public-methods, line-too-long, protected-access
from pathlib import Path
from inspect import getsourcefile
import unittest
import numpy as np

from ebop_maven.estimator import Estimator
from ebop_maven.tensorflow_models import load_model
from ebop_maven.libs import deb_example

class TestEstimator(unittest.TestCase):
    """ Unit tests for the TensorFlowEstimator base class. """
    # Tests for tf.DataSet support are in test_tensorflow_estimator_dataset
    _this_dir = Path(getsourcefile(lambda:0)).parent
    _default_model_file = _this_dir / "data/test_estimator/cnn_ext_model.keras"

    #
    #   TEST __init__(self, Model | Path)
    #
    def test_init_model_argument_errors(self):
        """ Tests __init__(invalid model values) -> correct error raised """
        self.assertRaises(ValueError, Estimator, None, 1)
        self.assertRaises(ValueError, Estimator, self._this_dir / "what.h5", 1)
        self.assertRaises(TypeError, Estimator, "I'm not a Path or Model", 1)

        valid_file = self._this_dir / "cnn_ext_model.h5"
        self.assertRaises(TypeError, Estimator, valid_file, iterations=None)
        self.assertRaises(ValueError, Estimator, valid_file, iterations=0)

    def test_init_with_path(self):
        """ Tests __init__(valid model path) -> correct initialization """
        estimator = Estimator(self._default_model_file)
        self.assertIn("CNN-Ext-Estimator", estimator.name)

    def test_init_with_model(self):
        """ Tests __init__(valid model path) -> correct initialization """
        # Not tf.keras.models.load_model as we're potentially using custom layers
        my_model = load_model(self._default_model_file)
        estimator = Estimator(my_model)
        self.assertIn("CNN-Ext-Estimator", estimator.name)

    #
    #   TEST predict(self, instances = [(mags_feature, ext_features)])
    #   NOTE: not testing the predictions made, just that the "plumbing" works
    #
    def test_predict_invalid_arguments(self):
        """ Tests predict(various invalid arg types) gives TypeError """
        estimator = Estimator(self._default_model_file)
        self.assertRaises(TypeError, estimator.predict, None)
        self.assertRaises(TypeError, estimator.predict, 2112)

    def test_predict_missing_instance_lc_item(self):
        """ Tests predict(missing the mandatory lc items) gives KeyError """
        estimator = Estimator(self._default_model_file)
        # only lc is mandatory
        self.assertRaises(KeyError, estimator.predict, [{ "dS_over_dP": 1, "phiS": 0.5 }])
        # Nothing else is
        instances = [{ "lc": [0.5] * estimator.lc_feature_bins }]
        estimator.predict(instances)

    def test_predict_incorrect_sized_instance_lc_bins(self):
        """ Tests predict(incorrect size lc item) gives ValueError """
        estimator = Estimator(self._default_model_file)
        instances = [{"lc": [0.5] * (estimator.lc_feature_bins+1) }]
        self.assertRaises(ValueError, estimator.predict, instances)

    def test_predict_instances_as_single_dict(self):
        """ Tests predict({ instance items }) handles dict and makes single prediction """
        estimator = Estimator(self._default_model_file)
        instances = { "lc": [0.5] * estimator.lc_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        result = estimator.predict(instances)
        self.assertEqual(len(result), 1)

    def test_predict_instances_as_list(self):
        """ Tests predict([{ instance items }]) makes correct number of predictions """
        estimator = Estimator(self._default_model_file)
        instances = [
            { "lc": [0.5] * estimator.lc_feature_bins, "dS_over_dP": 1, "phiS": 0.5 },
            { "lc": [0.25] * estimator.lc_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        ]
        result = estimator.predict(instances)
        self.assertEqual(len(result), len(instances))

    def test_predict_instances_as_ndarray(self):
        """ Tests predict([{ instance items }]) makes correct number of predictions """
        estimator = Estimator(self._default_model_file)
        instances = np.array([
            { "lc": [0.5] * estimator.lc_feature_bins, "dS_over_dP": 1, "phiS": 0.5 },
            { "lc": [0.25] * estimator.lc_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        ])
        result = estimator.predict(instances)
        self.assertEqual(len(result), len(instances))

    def test_predict_no_mc_dropout(self):
        """ Tests predict({ instance items }) without MC Dropout -> expect all sigmas==0 """
        estimator = Estimator(self._default_model_file, 1)
        instances = { "lc": [0.5] * estimator.lc_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        result = estimator.predict(instances)
        self.assertTrue(all(result[0][c]==0 for c in deb_example.label_predict_cols if c.endswith("_sigma")))

    def test_predict_with_mc_dropout(self):
        """ Tests predict({ instance items }) with MC Dropout -> expect some sigmas!=0 """
        estimator = Estimator(self._default_model_file, 100)
        instances = { "lc": [0.5] * estimator.lc_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        result = estimator.predict(instances)
        self.assertTrue(any(result[0][c]!=0 for c in deb_example.label_predict_cols if c.endswith("_sigma")))

if __name__ == "__main__":
    unittest.main()
