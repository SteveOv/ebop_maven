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
from ebop_maven import modelling, deb_example

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
        self.assertIn("rA_plus_rB", estimator.label_names)

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
    #   TEST predict(self, instances=List[Dict], iterations=int, unscale=bool, seed=int)
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
        # only mag is mandatory
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


    #
    #   TEST predict_raw(self, mags_feature=[#insts, #bins, 1], extra_features=[#insts, #feats, 1], iterations=int, unscale=bool, seed=int)
    #   NOTE: not testing the predictions made, just that the "plumbing" works
    #
    def test_predict_raw_invalid_argument_types(self):
        """ Tests predict(various invalid arg types) gives TypeError """
        estimator = Estimator(self._default_model_file)
        self.assertRaises(TypeError, estimator.predict_raw, None, np.array([1.0, 0.5]))
        self.assertRaises(TypeError, estimator.predict_raw, "Hello", np.array([1.0, 0.5]))
        self.assertRaises(TypeError, estimator.predict_raw, np.array([0.5]*1024), None)
        self.assertRaises(TypeError, estimator.predict_raw, np.array([0.5]*1024), "World")

    def test_predict_raw_inconsistent_inst_counts(self):
        """ Tests predict(inconsistent lengths on mags_feature and extra_features) gives ValueError """
        estimator = Estimator(self._default_model_file)
        inst_mags = [0.5] * estimator.mags_feature_bins
        inst_feats = list(range(len(estimator.input_feature_names)-1))
        mags_feature = np.array([inst_mags])                    # Shape (1, #bins)
        extra_features = np.array([inst_feats, inst_feats])     # Shape (2, #feats)
        self.assertRaises(ValueError, estimator.predict_raw, mags_feature, extra_features)

    def test_predict_raw_incorrect_sized_mags_feature(self):
        """ Tests predict(incorrect size mags_feature) gives ValueError """
        estimator = Estimator(self._default_model_file)
        inst_mags = [0.5] * (estimator.mags_feature_bins+1)
        inst_feats = list(range(len(estimator.input_feature_names)-1))
        mags_feature = np.array([inst_mags])                    # Shape (1, not #bins)
        extra_features = np.array([inst_feats])                 # Shape (1, #feats)
        self.assertRaises(ValueError, estimator.predict_raw, mags_feature, extra_features)

    def test_predict_raw_incorrect_sized_extra_features(self):
        """ Tests predict(incorrect size extra_features) gives ValueError """
        estimator = Estimator(self._default_model_file)
        inst_mags = [0.5] * estimator.mags_feature_bins
        inst_feats = list(range(len(estimator.input_feature_names)+1))
        mags_feature = np.array([inst_mags])                    # Shape (1, #bins)
        extra_features = np.array([inst_feats])                 # Shape (1, not #feats)
        self.assertRaises(ValueError, estimator.predict_raw, mags_feature, extra_features)

    def test_predict_raw_correctly_shaped_features(self):
        """ Tests predict(incorrect size extra_features) gives ValueError """
        estimator = Estimator(self._default_model_file)
        inst_mags = [0.5] * estimator.mags_feature_bins
        inst_feats = list(range(len(estimator.input_feature_names)-1))

        mags_feature = np.array([inst_mags])                    # Shape (1, #bins)
        extra_features = np.array([inst_feats])                 # Shape (1, #feats)
        preds0 = estimator.predict_raw(mags_feature, extra_features, iterations=1)

        mags_feature = mags_feature[:, :, np.newaxis]           # Shape (1, #bins, 1)
        extra_features = extra_features[:, :, np.newaxis]       # Shape (1, #feats, 1)
        preds1 = estimator.predict_raw(mags_feature, extra_features, iterations=1)
        self.assertEqual(preds0.tolist(), preds1.tolist())

    def test_predict_raw_with_explicit_iterations(self):
        """ Tests predict(_, _, iterations) expect result as (#insts, #labels, #iterations) """
        estimator = Estimator(self._default_model_file)
        inst_mags = np.array([[0.5] * estimator.mags_feature_bins])
        inst_feats = np.array([list(range(len(estimator.input_feature_names)-1))])
        for iterations in [None, 1, 100, 250]: # Expect it to coalesce None to 1
            result = estimator.predict_raw(inst_mags, inst_feats, iterations=iterations)
            self.assertEqual(iterations or 1, result.shape[2])

    def test_predict_raw_with_default_iterations(self):
        """ Tests predict({ instance items }, 100 MC Dropout iterations) -> expect some sigmas!=0 """
        iterations = 42
        estimator = Estimator(self._default_model_file, iterations=iterations)
        inst_mags = np.array([[0.5] * estimator.mags_feature_bins])
        inst_feats = np.array([list(range(len(estimator.input_feature_names)-1))])
        result = estimator.predict_raw(inst_mags, inst_feats)
        self.assertEqual(iterations, result.shape[2])

    def test_predict_raw_with_unscale(self):
        """ Tests predict({ instance items }, unscale=True) -> expect unscaled inc """
        estimator = Estimator(self._default_model_file)
        inst_mags = np.array([[0.5] * estimator.mags_feature_bins])
        inst_feats = np.array([list(range(len(estimator.input_feature_names)-1))])

        result = estimator.predict_raw(inst_mags, inst_feats, iterations=1, unscale=True)
        self.assertTrue(result[0][estimator.label_names.index("inc")] > 5, "expecting unscaled inc")

        result = estimator.predict_raw(inst_mags, inst_feats, iterations=1, unscale=False)
        self.assertTrue(result[0][estimator.label_names.index("inc")] < 5, "expecting scaled inc")



    #
    #   TEST means_and_stddevs_from_predictions(self, predictions: NDArray, labels_axis: int)
    #   NOTE: not testing the predictions made, just that the "plumbing" works
    #
    def test_means_and_stddevs_from_predictions_invalid_type(self):
        """ Tests means_and_stddevs_from_predictions(various invalid arg types) gives TypeError """
        estimator = Estimator(self._default_model_file)
        preds = np.ones(shape=(1, 6, 100))
        self.assertRaises(TypeError, estimator.means_and_stddevs_from_predictions, None, 1)
        self.assertRaises(TypeError, estimator.means_and_stddevs_from_predictions, 100, 1)
        self.assertRaises(TypeError, estimator.means_and_stddevs_from_predictions, preds, "H")

    def test_means_and_stddevs_from_predictions_infer_labels_axis(self):
        """
        Tests means_and_stddevs_from_predictions(various pred shapes) 
                                            -> infers label axis and returns expected shape result
        """
        estimator = Estimator(self._default_model_file)
        lab_count = len(estimator.label_names)
        for (preds, exp_shape) in [
            # (#labels,) with infered #insts==1 and #iters==1 -> (#means & #errors, )
            (np.ones(shape=lab_count), (lab_count*2, )),

            # (#labels, #iters) with infered #insts==1 -> (#means & #errors, )
            (np.ones(shape=(lab_count, 100)), (lab_count*2, )),

            # (#insts, #labels,) with infered #iters==1 -> (#insts, #means & #errors, )
            (np.ones(shape=(3, lab_count)), (3, lab_count*2)),

            # (#insts, #labels, #iters) -> (#insts, #means & #errors, )
            (np.ones(shape=(5, lab_count, 100)), (5, lab_count*2)),
        ]:
            result = estimator.means_and_stddevs_from_predictions(preds, label_axis=None)
            print(f"preds.shape=={preds.shape} -> result.shape=={result.shape} (exp=={exp_shape})")
            self.assertEqual(exp_shape, result.shape)

    def test_means_and_stddevs_from_predictions_explicit_labels_axis(self):
        """
        Tests means_and_stddevs_from_predictions(various pred shapes) 
                                            -> infers label axis and returns expected shape result
        """
        estimator = Estimator(self._default_model_file)

        for (preds, label_axis, exp_shape) in [
            (np.ones(shape=(4, 6)), 0, (8, )),
            (np.ones(shape=(4, 6)), 1, (4, 12)),

            (np.ones(shape=(2, 4, 6)), 1, (2, 8)),
        ]:
            result = estimator.means_and_stddevs_from_predictions(preds, label_axis)
            print(f"preds.shape=={preds.shape} -> result.shape=={result.shape} (exp=={exp_shape})")
            self.assertEqual(exp_shape, result.shape)

    def test_means_and_stddevs_from_predictions_assert_calcs_vsimple_preds(self):
        """
        Tests means_and_stddevs_from_predictions(implied 1 inst, 1 iter (#labels, ) preds) -> correct results
        """
        estimator = Estimator(self._default_model_file)

        # V-Simple (#insts==1, #labels==3, #iters==1) giving shape=(3,1)->
        #               [val(0), val(1), val(2), 0, 0, 0]
        full_preds = np.array([[0.3], [1.5], [5.4]])
        for preds in [
                full_preds,             # shape == (3, 1)
                full_preds.squeeze()    # shape == (3, )
            ]:
            result = estimator.means_and_stddevs_from_predictions(preds, label_axis=0)
            self.assertEqual((6,), result.shape)
            self.assertAlmostEqual(preds[0], result[0], 6)
            self.assertAlmostEqual(0, result[5], 6)

    def test_means_and_stddevs_from_predictions_assert_calcs_simple_preds(self):
        """
        Tests means_and_stddevs_from_predictions(implied 1 inst (#labels, #iters) preds) -> correct results
        """
        estimator = Estimator(self._default_model_file)

        # Simple; (#insts==1, #labels==3, #iters==5) giving shape=(3, 5) ->
        #               [mean(0,:), mean(1,:), mean(2,:), std(0,:), std(1,:), std(2,:)]
        preds = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.3, 1.5, 1.7, 1.8], [5.0, 5.2, 5.4, 5.6, 5.8]]
        )
        result = estimator.means_and_stddevs_from_predictions(preds, label_axis=0)
        self.assertEqual((6,), result.shape)
        self.assertAlmostEqual(np.mean(preds[0, :]), result[0], 6)
        self.assertAlmostEqual(np.std(preds[1, :]), result[4], 6)
        self.assertAlmostEqual(np.mean(preds[2, :]), result[2], 6)

    def test_means_and_stddevs_from_predictions_assert_calcs_full_preds(self):
        """
        Tests means_and_stddevs_from_predictions(full (#insts, #labels, #iters) preds) -> correct results
        """
        estimator = Estimator(self._default_model_file)

        # Less simple; (#insts==2, #labels==3, #iters==5) giving shape=(2, 3, 5) ->
        #           [
        #               [mean(0,0,:), mean(0,1,:), mean(0,2,:), std(0,0,:), std(0,1,:), std(0,2,:)],
        #               [mean(1,0,:), mean(1,1,:), mean(1,2,:), std(1,0,:), std(1,1,:), std(1,2,:)],
        #           ]
        preds = np.array([
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.3, 1.5, 1.7, 1.8], [5.0, 5.2, 5.4, 5.6, 5.8]],
            [[5.1, 6.1, 7.1, 8.1, 9.1], [15.0, 16.0, 17.0, 18.0, 19.0], [2.2, 2.4, 2.6, 2.8, 3.0]],
        ])
        result = estimator.means_and_stddevs_from_predictions(preds, label_axis=1)
        self.assertEqual((2, 6), result.shape)
        self.assertAlmostEqual(np.mean(preds[0, 0, :]), result[0, 0], 6)
        self.assertAlmostEqual(np.std(preds[0, 2, :]), result[0, 5], 6)
        self.assertAlmostEqual(np.mean(preds[1, 2, :]), result[1, 2], 6)
        self.assertAlmostEqual(np.std(preds[1, 1, :]), result[1, 4], 6)


if __name__ == "__main__":
    unittest.main()
