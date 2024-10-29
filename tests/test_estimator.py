""" Unit tests for the TensorFlowEstimator base class. """
# pylint: disable=too-many-public-methods, line-too-long, protected-access
from pathlib import Path
from contextlib import redirect_stdout
from inspect import getsourcefile
import unittest

import numpy as np
from uncertainties import unumpy, UFloat

import tensorflow as tf
from keras import layers

from ebop_maven.libs.tee import Tee
from ebop_maven import modelling, deb_example

from ebop_maven.estimator import Estimator


class TestEstimator(unittest.TestCase):
    """ Unit tests for the TensorFlowEstimator base class. """
    # Tests for tf.DataSet support are in test_tensorflow_estimator_dataset
    _this_dir = Path(getsourcefile(lambda:0)).parent
    _test_model_file = _this_dir / "data/test_estimator/test_cnn_ext_model.keras"
    _test_model_name = "Test-CNN-Ext-Model"

    @classmethod
    def setUpClass(cls):
        """ Make sure the test model file exists. """

        # To re-generate the test model file delete the existing file (under data) & run these tests
        # TODO: issue #64 raised to create training dataset specific to tests
        model_file = cls._test_model_file
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
                        layers.Dropout(0.1),
                        layers.Dense(32, "leaky_relu")
                    ],
                    name=cls._test_model_name
                )

                model.summary()
                model.compile(loss=["mae"], optimizer="adam", metrics=["mse"])

                # These are created by make_synthetic_test_dataset.py (smaller than full trainset)
                files = list(Path("./datasets/synthetic-mist-tess-dataset").glob("**/*.tfrecord"))
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
        self.assertRaises(FileNotFoundError, Estimator, self._this_dir / "what.h5", 1)
        self.assertRaises(TypeError, Estimator, "I'm not a Path or Model", 1)

        valid_file = self._this_dir / "cnn_ext_model.h5"
        self.assertRaises(TypeError, Estimator, valid_file, iterations=None)
        self.assertRaises(ValueError, Estimator, valid_file, iterations=0)

    def test_init_with_none(self):
        """ Tests __init__(model is None) -> inits with default ./data/estimator/*.keras model """
        estimator = Estimator()
        self.assertIn("rA_plus_rB", estimator.label_names)

    def test_init_with_path(self):
        """ Tests __init__(valid model path) -> correct initialization """
        estimator = Estimator(self._test_model_file)
        self.assertIn("Test", estimator.name)

    def test_init_with_model(self):
        """ Tests __init__(valid model path) -> correct initialization """
        # Not tf.keras.models.load_model as we're potentially using custom layers
        my_model = modelling.load_model(self._test_model_file)
        estimator = Estimator(my_model)
        self.assertIn("Test", estimator.name)

    #
    #   TEST predict(self, mags_features:NDArray, [ext_features=NDArray, iterations:int, unscale:bool, include_raw: bool, seed:int])
    #   NOTE: not testing the predictions made, just that the "plumbing" works
    #
    def test_predict_invalid_argument_types(self):
        """ Tests predict(various invalid arg types) gives TypeError """
        estimator = Estimator(self._test_model_file)
        self.assertRaises(TypeError, estimator.predict, None, np.array([1.0, 0.5]))
        self.assertRaises(TypeError, estimator.predict, "Hello", np.array([1.0, 0.5]))
        self.assertRaises(TypeError, estimator.predict, np.array([[0.5]*estimator.mags_feature_bins]), 0.5)

    def test_predict_incorrect_sized_mags_feature(self):
        """ Tests predict(incorrect size mags_feature NDArray) gives ValueError """
        estimator = Estimator(self._test_model_file)
        ext_features = np.array([[1.0] * len(estimator.extra_feature_names)]) # valid

        mags_feature = np.array([0.5] * (estimator.mags_feature_bins))      # (#bins) no inst dimension
        self.assertRaises(ValueError, estimator.predict, mags_feature, ext_features)
        mags_feature = np.array([[0.5] * (estimator.mags_feature_bins + 2)]) # too wide
        self.assertRaises(ValueError, estimator.predict, mags_feature, ext_features)

    def test_predict_incorrect_sized_ext_features(self):
        """ Tests predict(incorrect extra_features NDArray) gives ValueError """
        estimator = Estimator(self._test_model_file)
        mags_feature = np.array([[0.5] * (estimator.mags_feature_bins)])    # valid

        ext_features = np.array([1.0] * len(estimator.extra_feature_names)) # (#feats) no inst dimension
        self.assertRaises(ValueError, estimator.predict, mags_feature, ext_features)
        ext_features = np.array([[1.0] * (len(estimator.extra_feature_names)+1)]) # too wide
        self.assertRaises(ValueError, estimator.predict, mags_feature, ext_features)

    def test_predict_valid_assert_scaling(self):
        """ Tests predict() assert scaling correctly applied """
        estimator = Estimator(self._test_model_file)

        mags_feature = np.array([[0.5] * (estimator.mags_feature_bins)])
        ext_features = np.array([[1.0, 0.5]])

        # Fudge factors for approx_equal to overcome prediction variability
        fudges = { k: max(0.1/v, 0.25) for k, v in estimator._labels_and_scales.items() }

        def approx_equal(a: UFloat, b: UFloat, fudge: float=1.0) -> bool:
            return abs(a.nominal_value - b.nominal_value) < fudge

        # Our control preds; made without scaling applied then manually scaled
        expected_preds = estimator.predict(mags_feature, ext_features, iterations=1, unscale=False)
        for name in expected_preds.dtype.names:
            expected_preds[0][name] /= estimator._labels_and_scales[name]

        # Test the predictions made Without MC Dropout
        preds = estimator.predict(mags_feature, ext_features, iterations=1, unscale=True)
        for name in estimator.label_names:
            exp, pred, fudge = expected_preds[0][name], preds[0][name], fudges[name]
            self.assertTrue(approx_equal(exp, pred, fudge),
                            f"Expected pred[{name}]={pred} to be within {fudge} of {exp}")

        # With MC Dropout. We're going to use the same number of instances and MC iterations as the
        # number of labels. This will allow scaling to be applied to the wrong axis, and if it is we
        # can detect it has happened as the inc label (scale 0.01) will apply where not expected.
        num_labels = len(estimator.label_names)
        mags_feature = np.array(mags_feature.tolist() * num_labels)
        ext_features = np.array(ext_features.tolist() * num_labels)
        preds = estimator.predict(mags_feature, ext_features, iterations=num_labels, unscale=True)
        for inst in range(num_labels):
            for name in estimator.label_names:
                exp, pred, fudge = expected_preds[0][name], preds[inst][name], fudges[name]
                self.assertTrue(approx_equal(exp, pred, fudge),
                            f"Expected pred[{inst}][{name}]={pred} to be within {fudge} of {exp}")

    def test_predict_valid_single_inst_assert_structure(self):
        """ Tests predict((1, #bins), (1, #feats), iterations=1) returns correctly structured result """
        estimator = Estimator(self._test_model_file)

        mags_feature = np.array([[0.5] * (estimator.mags_feature_bins)])
        ext_features = np.array([[1.0, 0.5]])
        preds = estimator.predict(mags_feature, ext_features, iterations=1)

        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.shape, (1, ))
        self.assertListEqual(list(preds.dtype.names), estimator.label_names)

    def test_predict_valid_single_inst_include_raw_preds_assert_structure(self):
        """ Tests predict((1, #bins), (1, #feats), iterations=10, include_raw_preds=True) returns correctly structured result """
        estimator = Estimator(self._test_model_file)

        mags_feature = np.array([[0.5] * (estimator.mags_feature_bins)])
        ext_features = np.array([[1.0, 0.5]])
        result = estimator.predict(mags_feature, ext_features, iterations=10, include_raw_preds=True)

        self.assertIsInstance(result, tuple)
        (_, raw_preds) = result
        self.assertIsInstance(raw_preds, np.ndarray)
        self.assertEqual(raw_preds.shape, (1, len(estimator.label_names), 10))

    def test_predict_iterations_1_assert_zero_error_bars(self):
        """ Tests predict((1, #bins), (1, #feats), iterations=1) results have error bars of zero """
        estimator = Estimator(self._test_model_file)

        mags_feature = np.array([[0.5] * (estimator.mags_feature_bins)])
        ext_features = np.array([[1.0, 0.5]])
        preds = estimator.predict(mags_feature, ext_features, iterations=1)

        error_bars = unumpy.std_devs(preds.tolist()).flatten()
        self.assertTrue(all(eb== 0 for eb in error_bars))

    def test_predict_iterations_100_assert_nonzero_error_bars(self):
        """ Tests predict((1, #bins), (1, #feats), iterations=100) results have error bars """
        estimator = Estimator(self._test_model_file)

        mags_feature = np.array([[0.5] * (estimator.mags_feature_bins)])
        ext_features = np.array([[1.0, 0.5]])
        preds = estimator.predict(mags_feature, ext_features, iterations=100)

        error_bars = unumpy.std_devs(preds.tolist()).flatten()
        self.assertTrue(any(eb != 0 for eb in error_bars))

    def test_predict_iterations_assert_force_repeatable_results(self):
        """ 
        Tests predict() with iterations and a reset dropout seed gives repeatable results.
        A PoC that it's possible to force repeatability which may be needed for test predictions.
        """
        estimator = Estimator(self._test_model_file, iterations=100)
        mags_feature = np.array([[0.5] * (estimator.mags_feature_bins)])
        ext_features = np.array([[1.0, 0.5]])

        self._force_reset_dropout_seed(estimator, 2112)
        preds1 = estimator.predict(mags_feature, ext_features)[0]

        self._force_reset_dropout_seed(estimator, 2112)
        preds2 = estimator.predict(mags_feature, ext_features)[0]

        for ix, (p1, p2) in enumerate(zip(preds1, preds2, strict=True)):
            self.assertEqual(p1.nominal_value, p2.nominal_value, f"Index {ix} differs on nominals")
            self.assertEqual(p1.std_dev, p2.std_dev, f"Index {ix} differs on std_dev")

    @classmethod
    def _force_reset_dropout_seed(cls, estimator: Estimator, seed_value: int):
        """
        Messes with the seed generator used by each dropout layer to force a new state on it.
        Definitely not for "live" but may be useful for testing where repeatability is required. 
        """
        for layer in estimator._model.layers:
            if isinstance(layer, layers.Dropout):
                seed_gen = layer.seed_generator
                new_seed = seed_gen.backend.convert_to_tensor(np.array([0, seed_value],
                                                              dtype=seed_gen.state.dtype))
                seed_gen.state.assign(new_seed)

if __name__ == "__main__":
    unittest.main()
