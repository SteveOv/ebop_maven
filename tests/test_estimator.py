""" Unit tests for the TensorFlowEstimator base class. """
# pylint: disable=too-many-public-methods, line-too-long, protected-access
from pathlib import Path
from contextlib import redirect_stdout
from inspect import getsourcefile
import unittest

import numpy as np
import tensorflow as tf
from keras import layers

from ebop_maven.libs.tee import Tee
from ebop_maven import modelling
from ebop_maven.libs import deb_example

from ebop_maven.estimator import Estimator


class TestEstimator(unittest.TestCase):
    """ Unit tests for the TensorFlowEstimator base class. """
    # Tests for tf.DataSet support are in test_tensorflow_estimator_dataset
    _this_dir = Path(getsourcefile(lambda:0)).parent
    _default_model_file = _this_dir / "data/test_estimator/test_cnn_ext_model.keras"
    _default_model_name = "Test-CNN-Ext-Model"

    @classmethod
    def setUpClass(cls):
        """ Make sure the test model file exists. """
        # To re-generate the test model file delete the existing file (under data) & run these tests
        model_file = cls._default_model_file
        if not model_file.exists():
            with redirect_stdout(Tee(open(model_file.parent / f"{model_file.stem}.txt",
                                          "w",
                                          encoding="utf8"))):
                # A very simple compatible model
                model = modelling.build_mags_ext_model(
                    mags_layers=[
                        layers.Conv1D(32, 16, 4, "same", activation="relu"),
                        layers.AveragePooling1D(pool_size=4, strides=4, padding="same"),
                        layers.Conv1D(32, 16, 4, "same", activation="relu"),
                    ],
                    dnn_layers=[
                        layers.Dense(32, "leaky_relu")
                    ],
                    name=cls._default_model_name
                )

                model.summary()
                model.compile(loss=["mae"], optimizer="adam", metrics=["mse"])

                # These are created by make_training_datasets.py
                files = list(Path("./datasets/formal-training-dataset/training").glob("**/*.tfrecord"))
                train_ds = tf.data.TFRecordDataset(files, num_parallel_reads=100)
                x = train_ds.shuffle(100000, 42).map(deb_example.create_map_func()).batch(100)
                model.fit(x, epochs=10, verbose=2)

                model_file.parent.mkdir(parents=True, exist_ok=True)
                modelling.save_model(model_file, model)
        return super().setUpClass()

    #
    #   TEST __init__(self, Model | Path)
    #
    def test_init_model_argument_errors(self):
        """ Tests __init__(invalid model values) -> correct error raised """
        self.assertRaises(ValueError, Estimator, self._this_dir / "what.h5", 1)
        self.assertRaises(TypeError, Estimator, "I'm not a Path or Model", 1)

        valid_file = self._this_dir / "cnn_ext_model.h5"
        self.assertRaises(TypeError, Estimator, valid_file, iterations=None)
        self.assertRaises(ValueError, Estimator, valid_file, iterations=0)

    def test_init_with_none(self):
        """ Tests __init__(model is None) -> inits with default ./data/estimator/*.keras model """
        estimator = Estimator()
        self.assertIn("Estimator", estimator.name)

    def test_init_with_path(self):
        """ Tests __init__(valid model path) -> correct initialization """
        estimator = Estimator(self._default_model_file)
        self.assertIn("Test", estimator.name)

    def test_init_with_model(self):
        """ Tests __init__(valid model path) -> correct initialization """
        # Not tf.keras.models.load_model as we're potentially using custom layers
        my_model = modelling.load_model(self._default_model_file)
        estimator = Estimator(my_model)
        self.assertIn("Test", estimator.name)

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
        instances = [{ "mags": [0.5] * estimator.mags_feature_bins }]
        estimator.predict(instances)

    def test_predict_incorrect_sized_instance_lc_bins(self):
        """ Tests predict(incorrect size lc item) gives ValueError """
        estimator = Estimator(self._default_model_file)
        instances = [{"mags": [0.5] * (estimator.mags_feature_bins+1) }]
        self.assertRaises(ValueError, estimator.predict, instances)

    def test_predict_instances_as_single_dict(self):
        """ Tests predict({ instance items }) handles dict and makes single prediction """
        estimator = Estimator(self._default_model_file)
        instances = { "mags": [0.5] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        result = estimator.predict(instances)
        self.assertEqual(len(result), 1)

    def test_predict_instances_as_list(self):
        """ Tests predict([{ instance items }]) makes correct number of predictions """
        estimator = Estimator(self._default_model_file)
        instances = [
            { "mags": [0.5] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 },
            { "mags": [0.25] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        ]
        result = estimator.predict(instances)
        self.assertEqual(len(result), len(instances))

    def test_predict_instances_as_ndarray(self):
        """ Tests predict([{ instance items }]) makes correct number of predictions """
        estimator = Estimator(self._default_model_file)
        instances = np.array([
            { "mags": [0.5] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 },
            { "mags": [0.25] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        ])
        result = estimator.predict(instances)
        self.assertEqual(len(result), len(instances))

    def test_predict_no_mc_dropout(self):
        """ Tests predict({ instance items }) without MC Dropout -> expect all sigmas==0 """
        estimator = Estimator(self._default_model_file, 1)
        instances = { "mags": [0.5] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        result = estimator.predict(instances)
        self.assertTrue(all(result[0][c]==0 for c in estimator.prediction_names if c.endswith("_sigma")))

    def test_predict_with_mc_dropout(self):
        """ Tests predict({ instance items }) with MC Dropout -> expect some sigmas!=0 """
        estimator = Estimator(self._default_model_file, 100)
        instances = { "mags": [0.5] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }
        result = estimator.predict(instances)
        self.assertTrue(any(result[0][c]!=0 for c in estimator.prediction_names if c.endswith("_sigma")))

    def test_predict_with_iterations_overriding_default(self):
        """ Tests predict({ instance items }, 100 MC Dropout iterations) -> expect some sigmas!=0 """
        estimator = Estimator(self._default_model_file, iterations=1) # 1 iteration == no MC Dropout
        instances = { "mags": [0.5] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }

        # Make a prediction with 100 MC Dropout iterations - if this is ignored ours sigmas will all == 0
        result = estimator.predict(instances, iterations=100)
        self.assertTrue(any(result[0][c]!=0 for c in estimator.prediction_names if c.endswith("_sigma")))

    def test_predict_with_unscale_on(self):
        """ Tests predict({ instance items }, unscale=True) -> expect unscaled inc """
        estimator = Estimator(self._default_model_file, iterations=1) # 1 iteration == no MC Dropout
        instances = { "mags": [0.5] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }

        result = estimator.predict(instances, unscale=True)
        self.assertTrue(result[0]["inc"] > 5, "expecting inc to not be re-scaled")

    def test_predict_with_unscale_off(self):
        """ Tests predict({ instance items }, unscale=False) -> expect scaled inc """
        estimator = Estimator(self._default_model_file, iterations=1) # 1 iteration == no MC Dropout
        instances = { "mags": [0.5] * estimator.mags_feature_bins, "dS_over_dP": 1, "phiS": 0.5 }

        result = estimator.predict(instances, unscale=False)
        self.assertTrue(result[0]["inc"] < 5, "expecting inc to be scaled value")

if __name__ == "__main__":
    unittest.main()
