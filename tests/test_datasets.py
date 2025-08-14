""" Tests for the deb_example module. """
from threading import Lock
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf
from deblib.orbital import impact_parameter

from ebop_maven import deb_example
from traininglib.datasets import create_map_func, create_dataset_pipeline
from traininglib.datasets import iterate_dataset, read_dataset
from traininglib.datasets import _swap_instance_components # pylint: disable=protected-access


# pylint: disable=invalid-name, too-many-public-methods, line-too-long, protected-access, too-many-locals, cell-var-from-loop
class Test_datasets(unittest.TestCase):
    """ Tests for the datasets module """
    # These tests may fiddle with the description so should not be run parallel
    lock = Lock()

    #
    #   TEST create_map_func(mags_bins: int, mags_wrap_phse: float,
    #                        ext_features: List[str], labels: List[str],
    #                        noise_stddev: Callable->float, roll_steps: Callable->int) -> Callable
    #
    def test_create_map_func_default_behaviour(self):
        """ Test that the resulting map_func accurately deserializes a deb_example """
        with self.__class__.lock:
            # Set up a feature (light-curve) amd labels and with tracable values
            input_labels = { k: v for v, k in enumerate(deb_example.labels_and_scales) }

            # The mags_feature source is a folded-lc twice the length of the resulting feature with
            # alternate mags values of 0 and 2. We expect each output bin should avg (0+2)/2 = 1.
            input_phases = np.flip(np.linspace(1.0, 0.0, deb_example.default_mags_bins * 2))
            input_mags = np.array([0.0, 2.0] * deb_example.default_mags_bins, dtype=float)
            exp_mag_feature = np.ones((deb_example.default_mags_bins), dtype=float)

            input_ext_features = { "phiS": 0.6, "dS_over_dP": 0.96 }
            deb = deb_example.serialize("t1", input_labels, input_phases, input_mags, None, input_ext_features)

            # Execute a graph instance of the map_func to mimic a Dateset pipeline.
            # map_parse_fn = create_map_func()
            map_parse_fn = tf.function(create_map_func())
            ((lc_feature, ext_features), labels) = map_parse_fn(deb)

            # lc output should be a Tensor of shape (default_mags_bins, 1)
            # with content the mean of bins taken from the input
            self.assertEqual(lc_feature.shape, (deb_example.default_mags_bins, 1))
            self.assertListEqual(lc_feature.numpy()[..., 0].tolist(), exp_mag_feature.tolist())

            # features output should be a Tensor of the shape (#features, 1)
            self.assertEqual(ext_features.shape, (len(deb_example.extra_features_and_defaults), 1))
            for feature, exp_value in zip(ext_features.numpy(), input_ext_features.values()):
                self.assertEqual(feature, exp_value)

            # labels output should be a list of Tensors of length #labels
            # Assert they have been scaled and are in correct order
            self.assertEqual(len(labels), len(deb_example.labels_and_scales))
            self.assertIsInstance(labels, list)
            exp_values = [v * s for v, s in zip(input_labels.values(), deb_example.labels_and_scales.values())]
            for label, exp_value in zip(labels, exp_values):
                # Some loss of fidelity in encoding/decoding a tf.Tensor so can't do exact assert
                self.assertAlmostEqual(label.numpy(), exp_value, 6)

    def test_create_map_func_wrap_phase(self):
        """ Tests the created map_func's wrap_phase functionality """
        with self.__class__.lock:
            # Phases/mags with each bin's value equal to its phase; makes tracking wrap easier.
            input_labels = { k: v for v, k in enumerate(deb_example.labels_and_scales) }
            mags_bins = deb_example.default_mags_bins
            input_phases = input_mags = np.flip(np.linspace(1.0, 0.0, mags_bins))

            # With wrap_phase we're specifying the last phase in the output; any input phased after
            # this value will "wrapped" round to a negative phase (unless it is zero, when it is
            # ignored). For example, if wrap_phase=0.75 we expect the output phases to cover
            # (-0.25, 0.75], effectively shifting the zero phase "right" by 25% of the data's length
            for wrap_phase, exp_phase_zero_ix in [(0.0, 0),
                                                  (1.0, 0),
                                                  (0.5, mags_bins * 0.5),
                                                  (0.25, mags_bins * 0.75),
                                                  (1.25, mags_bins * 0.75),
                                                  (0.75, mags_bins * 0.25),
                                                  (-0.25, mags_bins * 0.25)]:
                # pylint: disable=cell-var-from-loop
                # Execute a graph instance of the map_func, with wrap, to mimic a Dateset pipeline.
                map_parse_fn = tf.function(create_map_func(mags_wrap_phase=wrap_phase))

                deb = deb_example.serialize("t1", input_labels, input_phases, input_mags)
                ((lc_feature, _), _) = map_parse_fn(deb)

                # Asserting the phase zero (originally zeroth bin) is where we expect
                lc_feature = lc_feature.numpy()[:, 0]
                phase_zero_ix = np.where(lc_feature == 0)[0]
                self.assertEqual(phase_zero_ix, exp_phase_zero_ix)

    def test_create_map_func_adaptive_wrap_phase(self):
        """ Tests the created map_func's adaptive wrap_phase functionality """
        with self.__class__.lock:
            # Phases/mags with each bin's value equal to its phase; makes tracking wrap easier.
            input_labels = { k: v for v, k in enumerate(deb_example.labels_and_scales) }
            mags_bins = deb_example.default_mags_bins
            input_phases = input_mags = np.flip(np.linspace(1.0, 0.0, mags_bins))

            # A graph instance of the map_func, with wrap==None to enforce adaptive wrap.
            map_fn = tf.function(create_map_func(mags_bins, mags_wrap_phase=None))

            for phiS in [0.5, 0.2, 0.33, 0.66, 0.8]:
                deb = deb_example.serialize("t1", input_labels, input_phases, input_mags, None,
                                            extra_features={ "phiS": phiS })
                ((lc_feature, _), _) = map_fn(deb)

                # Work out where we expect the parsed mags to be centred; the midpoint between
                # the primary (at phase 0) and secondary (at phase phiS).
                eclipse_midpoint_phase = phiS / 2
                eclipse_midpoint_value = input_mags[int(mags_bins * eclipse_midpoint_phase)]

                # We expect the mags to now be centred on the midpoint between eclipses.
                centre_bin_value = lc_feature.numpy()[mags_bins // 2, 0]
                self.assertTrue(abs(eclipse_midpoint_value - centre_bin_value) <= 1)

    def test_create_map_func_with_selected_ext_features(self):
        """ Test that the resulting map_func deserializes a deb_example with subset of ext_features """
        with self.__class__.lock:
            # Set up a mags feature and labels and with tracable values
            input_labels_and_scales = deb_example.labels_and_scales.copy()
            input_labels = { k: v for v, k in enumerate(input_labels_and_scales) }
            input_phases = input_mags = np.flip(np.linspace(1., 0., deb_example.default_mags_bins))
            input_ext_features = { "phiS": 0.6, "dS_over_dP": 0.96 }
            deb = deb_example.serialize("t1", input_labels,
                                        input_mags, input_phases, None, input_ext_features)

            # We're going to request a shuffled subset of the ext_features
            request_features = [f for f in input_ext_features if f not in ["phiS"]]

            # Execute a graph instance of the map_func to mimic a Dateset pipeline.
            # map_parse_fn = reate_map_func(ext_features=request_features)
            map_parse_fn = tf.function(create_map_func(ext_features=request_features))
            ((mags_feature, ext_features), labels) = map_parse_fn(deb)

            # features output should be a Tensor of the shape (#request_features, 1)
            self.assertEqual(ext_features.shape, (len(request_features), 1))
            exp_values = [input_ext_features[e] for e in request_features]
            for feature, exp_value in zip(ext_features.numpy(), exp_values):
                self.assertEqual(feature, exp_value)

            # Check no effect on the other values
            self.assertEqual(mags_feature.shape, (deb_example.default_mags_bins, 1))
            self.assertEqual(len(labels), len(input_labels))

    def test_create_map_func_with_selected_labels(self):
        """ Test that the resulting map_func deserializes a deb_example with subset of labels """
        with self.__class__.lock:
            # Set up a feature (light-curve) amd labels and with tracable values
            input_labels_and_scales = deb_example.labels_and_scales.copy()
            input_labels = { k: v for v, k in enumerate(input_labels_and_scales) }
            input_phases = input_mags = np.flip(np.linspace(1., 0., deb_example.default_mags_bins))
            input_ext_features = { "phiS": 0.6, "dS_over_dP": 0.96 }
            deb = deb_example.serialize("t1", input_labels,
                                        input_mags, input_phases, None, input_ext_features)

            # We're going to request a shuffled subset - we should only get these and in this order
            request_labels = [k for k in input_labels_and_scales if k not in ["J"]]
            np.random.shuffle(request_labels)

            # Execute a graph instance of the map_func to mimic a Dateset pipeline.
            # map_parse_fn = create_map_func(labels=request_labels)
            map_parse_fn = tf.function(create_map_func(labels=request_labels))
            (_, labels) = map_parse_fn(deb)

            # labels output should be a list of Tensors of length #request_labels
            # Assert they have been scaled and are in request_labels order
            self.assertEqual(len(labels), len(request_labels))
            self.assertIsInstance(labels, list)
            exp_values = [input_labels[l] * input_labels_and_scales[l] for l in request_labels]
            for label, exp_value in zip(labels, exp_values):
                # Some loss of fidelity in encoding/decoding a tf.Tensor so can't do exact assert
                self.assertAlmostEqual(label.numpy(), exp_value, 6)

    def test_create_map_func_with_roll_augmentation(self):
        """ Tests the created map_func's augmentation functionality supports rolling the mags """
        with self.__class__.lock:
            # Set up a feature (light-curve) amd labels and with tracable values
            input_labels = { k: v for v, k in enumerate(deb_example.labels_and_scales) }
            input_phases = np.flip(np.linspace(1., 0., deb_example.default_mags_bins))
            input_mags = np.arange(deb_example.default_mags_bins, dtype=float)
            deb = deb_example.serialize("t1", input_labels, input_phases, input_mags)

            for roll_steps in [-5, 0, 5]:
                def aug_callback(mags_feature):
                    return tf.roll(mags_feature, [roll_steps], axis=[0])

                # pylint: disable=cell-var-from-loop
                # Execute a graph instance of the map_func (with roll) to mimic a Dateset pipeline.
                map_parse_fn = tf.function(create_map_func(augmentation_callback=aug_callback))
                ((lc_feature, _), _) = map_parse_fn(deb)

                # Assert that these bins match the input values where they should have been rolled from
                lc_feature = lc_feature.numpy()[:, 0]
                for lb_ix in np.arange(500, 600, 1):
                    self.assertEqual(lc_feature[lb_ix + roll_steps], input_mags[lb_ix])

    def test_create_map_func_with_mags_wrap_and_roll_augmentation(self):
        """ Tests the created map_func's mags_wrap and roll augmentations combine as expected """
        with self.__class__.lock:
            # Set up a feature (light-curve) amd labels and with tracable values
            # so the contents of each mags_bin matches its initial index with phase 0 at ix [0]
            input_labels = { k: v for v, k in enumerate(deb_example.labels_and_scales) }
            mags_bins = deb_example.default_mags_bins
            input_phases = np.flip(np.linspace(1., 0., mags_bins))
            input_mags = np.arange(mags_bins, dtype=float)
            deb = deb_example.serialize("t1", input_labels, input_phases, input_mags)

            # We expect the wrap_phase and roll_steps to be additive.
            # Wrap phase specifies the last phase of the output, with any input data phased beyond
            # this wrapped to negative phase (except if it is zero when it is ignored).
            # To this we add the roll_steps for augmentation/perturbation to the phase of the
            # datapoints, given directly as the number of bins to roll the data left or right.
            for wrap_phase, roll_steps, exp_phase_zero_ix in [(0.0, 0, 0),
                                                              (0.0, 5, 5),
                                                              (0.0, -5, mags_bins-5),
                                                              (1.0, 5, 5),
                                                              (1.0, -5, mags_bins-5),
                                                              (0.75, 10, (mags_bins*0.25)+10),
                                                              (0.75, -10, (mags_bins*0.25)-10),
                                                              (-0.25, -10, (mags_bins*0.25)-10),
                                                              (0.5, 15, (mags_bins*0.5)+15),
                                                              (0.25, -25, (mags_bins*0.75)-25)]:

                def aug_callback(mags_feature):
                    return tf.roll(mags_feature, [roll_steps], axis=[0])

                # pylint: disable=cell-var-from-loop
                # Execute a graph instance of the map_func, with wrap, to mimic a Dateset pipeline.
                map_parse_fn = tf.function(create_map_func(mags_wrap_phase=wrap_phase, augmentation_callback=aug_callback))
                ((lc_feature, _), _) = map_parse_fn(deb)

                # Asserting the phase zero (originally zeroth bin) is rolled to where we expect
                lc_feature = lc_feature.numpy()[:, 0]
                phase_zero_ix = np.where(lc_feature == 0)[0]
                self.assertEqual(phase_zero_ix, exp_phase_zero_ix)

    def test_create_map_func_with_noise_augmentation(self):
        """ Tests the created map_func's augmentation functionality supports adding noise """
        with self.__class__.lock:
            # Set up a feature (light-curve) amd labels and with tracable values
            input_labels = { k: v for v, k in enumerate(deb_example.labels_and_scales) }
            input_phases = np.flip(np.linspace(1., 0., deb_example.default_mags_bins))
            input_mags = np.ones_like(input_phases, dtype=float)
            deb = deb_example.serialize("t1", input_labels, input_phases, input_mags)

            apply_stddev = 0.01
            def aug_callback(mags_feature):
                return mags_feature + tf.random.normal(mags_feature.shape, stddev=apply_stddev)

            # Execute a graph instance of the map_func (with roll) to mimic a Dateset pipeline.
            map_parse_fn = tf.function(create_map_func(augmentation_callback=aug_callback))
            ((lc_feature, _), _) = map_parse_fn(deb)

            # Assert that stddev of the output lc reflects the noise we applied
            lc_feature = lc_feature.numpy()[:, 0]
            self.assertAlmostEqual(lc_feature.std(), apply_stddev, 3)

    def test_create_map_func_with_random_roll_augmentation(self):
        """ Tests the created map_func's augmentation functionality supports random rolls """
        with self.__class__.lock:
            # Set up a feature (light-curve) amd labels and with tracable values
            input_labels = { k: v for v, k in enumerate(deb_example.labels_and_scales) }
            input_phases = np.flip(np.linspace(1., 0., deb_example.default_mags_bins))
            input_mags = np.arange(deb_example.default_mags_bins, dtype=float)
            deb = deb_example.serialize("t1", input_labels, input_phases, input_mags)

            # Augmentation callback with random roll
            def aug_callback(mags_feature):
                roll_steps = tf.random.uniform([], -100, 101, tf.int32)
                return tf.roll(mags_feature, [roll_steps], axis=[0])

            map_parse_fn = tf.function(create_map_func(augmentation_callback=aug_callback))

            # Execute map_func a number of times on the same input lc with the random roll.
            # Store the output lcs, which should have different rolls applied to them.
            lc_copy_list = []
            for _ in range(5):

                ((lc_feature, _), _) = map_parse_fn(deb)
                lc_copy_list.append(lc_feature.numpy()[:, 0])

            # There should be variation per bin across the lcs due to random roll_steps value
            first_bins = np.vstack(lc_copy_list)[:, 0]
            print(f"first_bins == {first_bins}")
            self.assertFalse(all(bin == first_bins[0] for bin in first_bins), f"{first_bins}")

    #
    #   TEST iterate_dataset(dataset_files: Iterable[str], mags_bins: int, mags_wrap_phase: float,
    #                        ext_features: List[str], labels: List[str],
    #                        identifiers: List[str], scale_labels: bool,
    #                        noise_stddev: float, roll_max: int, max_instances: int)
    #                           -> Generator(ids, mags, features, labels)
    #
    #   iterate_dataset() wraps create_dataset_pipeline() so that gets a workout too
    #
    def test_iterate_dataset_default_args(self):
        """ Tests iterate_dataset(no id filter) -> all rows """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        (id_val, mrow, frow, lrow) = next(iterate_dataset(files))
        # Check the dimensions of whats returned for the first row; should reflect the defaults
        self.assertIsNotNone(id_val)
        self.assertEqual(deb_example.default_mags_bins, mrow.shape[0])
        self.assertEqual(len(deb_example.extra_features_and_defaults), frow.shape[0])
        self.assertEqual(len(deb_example.labels_and_scales), lrow.shape[0])

    def test_iterate_dataset_no_filters(self):
        """ Tests iterate_dataset(no id filter) -> all rows """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        (_, row_count) = create_dataset_pipeline(files)

        iterate_count = 0
        for _ in iterate_dataset(files, identifiers=None):
            iterate_count += 1
        self.assertEqual(row_count, iterate_count)

    def test_iterate_dataset_identifers_filtering(self):
        """ Tests iterate_dataset(with id filter) -> selected rows in stored order """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        identifiers = ["CM Dra", "CW Eri"] # Not in the order they appear in the dataset
        id_vals = []
        for (id_val, _, _, _) in iterate_dataset(files, identifiers=identifiers):
            id_vals += [id_val]
        # Only the two rows, yielded in the order they appear in the dataset
        self.assertEqual(["CW Eri", "CM Dra"], id_vals)

    def test_iterate_dataset_identifers_filtering_and_max_instances(self):
        """ Tests iterate_dataset(with id filter and max_instances) -> yields up to max_instances """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        exp_inst_count = 2
        identifiers = ["CM Dra", "CW Eri", "GW Eri", "RR Lyn"]
        id_vals = []
        for (id_val, _, _, _) in iterate_dataset(files, identifiers=identifiers, max_instances=exp_inst_count):
            id_vals += [id_val]
        self.assertEqual(exp_inst_count, len(id_vals))

    def test_iterate_dataset_filter_func_on_id(self):
        """ Tests iterate_dataset(with filter_func set) -> yields only instances matching filter """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        exp_id_vals = ["CW Eri"]
        id_vals = []

        @tf.function
        def filter_func(inst_id, feature_vals, label_vals): # pylint: disable=unused-argument
            return inst_id == "CW Eri"

        for (id_val, _, _, _) in iterate_dataset(files, filter_func=filter_func):
            id_vals += [id_val]
        self.assertListEqual(exp_id_vals, id_vals)

    def test_iterate_dataset_filter_func_on_extra_feature(self):
        """ Tests iterate_dataset(with filter_func set) -> yields only instances matching filter """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        exp_id_vals = ["MU Cas", "V362 Pav", "CW Eri", "AN Cam", "IT Cas"]
        id_vals = []
        ix_phiS = [*deb_example.extra_features_and_defaults.keys()].index("phiS")

        @tf.function
        def filter_func(inst_id, feature_vals, label_vals): # pylint: disable=unused-argument
            return feature_vals[1][ix_phiS][0] > 0.5

        for (id_val, _, _, _) in iterate_dataset(files, filter_func=filter_func):
            id_vals += [id_val]
        self.assertListEqual(exp_id_vals, id_vals)

    def test_iterate_dataset_filter_func_on_label(self):
        """ Tests iterate_dataset(with filter_func set) -> yields only instances matching filter """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        exp_id_vals = ["V436 Per", "V889 Aql"]
        id_vals = []
        ix_esinw = [*deb_example.labels_and_scales.keys()].index("esinw")

        @tf.function
        def filter_func(inst_id, feature_vals, label_vals): # pylint: disable=unused-argument
            return label_vals[ix_esinw] > 0.3

        for (id_val, _, _, _) in iterate_dataset(files, filter_func=filter_func):
            id_vals += [id_val]
        self.assertListEqual(exp_id_vals, id_vals)

    def test_iterate_dataset_max_instances_low(self):
        """ Tests iterate_dataset(max_instances < ds rows) -> return requested number of rows """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        exp_inst_count = 5
        id_vals = []
        for (id_val, _, _, _) in iterate_dataset(files, max_instances=exp_inst_count):
            id_vals += [id_val]
        self.assertEqual(exp_inst_count, len(id_vals))

    def test_iterate_dataset_max_instances_high(self):
        """ Tests iterate_dataset(max_instances > ds rows) -> return all availabe rows; no error """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        (_, exp_inst_count) = create_dataset_pipeline(files)
        id_vals = []
        for (id_val, _, _, _) in iterate_dataset(files, max_instances=100000):
            id_vals += [id_val]
        self.assertEqual(exp_inst_count, len(id_vals))

    def test_iterate_dataset_all_filters(self):
        """ Tests iterate_dataset(with id, features & labels filter and scaling) -> selected rows """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        (id_val, mrow, frow, lrow) = next(iterate_dataset(files, 4096, 1.0, ext_features=["phiS"], labels=["inc"],
                                                          identifiers=["CW Eri"], scale_labels=True))

        self.assertEqual("CW Eri", id_val)
        self.assertEqual(4096, len(mrow))
        self.assertEqual(1, len(frow))
        self.assertAlmostEqual(0.503, frow[0], 3)   # feature: phiS (dS_over_dP near 1.0)
        self.assertEqual(1, len(lrow))
        self.assertAlmostEqual(0.864, lrow[0], 3)   # label: inc scaled (so inc/100)


    #
    #   TEST read_dataset(dataset_files: Iterable[str], mags_bins: int, mags_wrap_phse: float,
    #                     ext_features: List[str], labels: List[str],
    #                     identifiers: List[str], scale_labels: bool,
    #                     noise_stddev: float, roll_max: int, max_instances: int)
    #                       -> (ids, mags, features, labels)
    #
    def test_read_dataset_identifers_filtering(self):
        """ Tests read_dataset(with id & label filters) -> selected rows in given order """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))

        identifiers = ["CM Dra", "CW Eri"] # Not in the order they appear in the dataset
        labels = ["L3", "inc"]
        (id_vals, _, _, lrows) = read_dataset(files, identifiers=identifiers, labels=labels,
                                              scale_labels=False, max_instances=199)

        # Only the two rows, yielded in the order requested, not the order they're stored in
        self.assertEqual(2, len(id_vals))
        self.assertEqual(identifiers[0], id_vals[0])        # CM Dra
        self.assertAlmostEqual(0, lrows[0]["L3"], 3)
        self.assertAlmostEqual(89.5514, lrows[0]["inc"], 3)
        self.assertEqual(identifiers[1], id_vals[1])        # CW Eri
        self.assertAlmostEqual(-0.0002, lrows[1]["L3"], 3)
        self.assertAlmostEqual(86.381, lrows[1]["inc"], 3)

    def test_read_dataset_labels_via_indices(self):
        """ Tests read_dataset() -> rows & values accessed via numeric indices (backward compat) """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))

        identifiers = ["CM Dra", "CW Eri"] # Not in the order they appear in the dataset
        (id_vals, _, _, lrows) = read_dataset(files, identifiers=identifiers)

        l3_ix = list(deb_example.labels_and_scales.keys()).index("L3")
        inc_ix = list(deb_example.labels_and_scales.keys()).index("inc")

        # Only the two rows, yielded in the order requested, not the order they're stored in
        self.assertEqual(2, len(id_vals))
        self.assertEqual(identifiers[0], id_vals[0])        # CM Dra
        self.assertAlmostEqual(0, lrows[0][l3_ix], 3)
        self.assertAlmostEqual(89.5514, lrows[0][inc_ix], 3)
        self.assertEqual(identifiers[1], id_vals[1])        # CW Eri
        self.assertAlmostEqual(-0.0002, lrows[1][l3_ix], 3)
        self.assertAlmostEqual(86.381, lrows[1][inc_ix], 3)

    def test_read_dataset_identifers_no_match(self):
        """ Tests read_dataset(with unknown id) -> no error but empty results returned """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))

        (id_vals, mags, feats, labs) = read_dataset(files, identifiers=["unknown"])

        # No rows
        self.assertEqual(id_vals.shape, (0, ))
        self.assertEqual(mags.shape, (0, ))
        self.assertEqual(feats.shape, (0, ))
        self.assertEqual(labs.shape, (0, ))
        self.assertEqual(len(labs["k"]), 0) # It knows the names but there is no data

    def test_read_dataset_identifers_no_labels(self):
        """ Tests read_dataset(with an unknown label) -> no error but empty labs returned """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))

        (id_vals, _, _, labs) = read_dataset(files, identifiers=["CW Eri", "CM Dra"], labels=[])
        # Two matching rows, but the labs rows are empty
        self.assertEqual(id_vals.shape, (2, ))
        self.assertEqual(labs.shape, (2, ))
        self.assertEqual(len(labs.dtype.names), 0)

    def test_read_dataset_identifers_max_instances(self):
        """ Tests read_dataset(with an unknown label) -> no error but empty labs returned """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        exp_inst_count = 5
        (id_vals, mags, feats, labs) = read_dataset(files, max_instances=exp_inst_count)

        # Only the requested number of rows have been returned
        self.assertEqual(id_vals.shape[0], exp_inst_count)
        self.assertEqual(mags.shape[0], exp_inst_count)
        self.assertEqual(feats.shape[0], exp_inst_count)
        self.assertEqual(labs.shape[0], exp_inst_count)

    def test_read_dataset_filter_func_on_feature(self):
        """ Tests read_dataset(with filter_func set) -> yields only instances matching filter """
        files = list((Path.cwd() / "datasets/formal-test-dataset/").glob("**/*.tfrecord"))
        exp_id_vals = ["MU Cas", "V362 Pav", "CW Eri", "AN Cam", "IT Cas"]
        id_vals = []
        ix_phiS = [*deb_example.extra_features_and_defaults.keys()].index("phiS")

        @tf.function
        def filter_func(inst_id, feature_vals, label_vals): # pylint: disable=unused-argument
            return feature_vals[1][ix_phiS][0] > 0.5

        (id_vals, _, _, _) = read_dataset(files, filter_func=filter_func)
        self.assertListEqual(exp_id_vals, id_vals.tolist())

    @unittest.skip("only run this interactively as it may take a long time")
    def test_read_dataset_scalability_test(self):
        """ Tests read_dataset(with large dataset) -> data is returned in seconds not minutes/hours """

        # There are approx 20,000 rows in this dataset - should be done in ~20 s
        # As read_dataset() wraps iterate_dataset() this tests both functions' scalability
        files = list((Path.cwd() / "datasets/synthetic-mist-tess-dataset/").glob("**/*.tfrecord"))
        (id_vals, _, _, lrows) = read_dataset(files, 4096, 0.75, scale_labels=False)
        self.assertTrue(len(id_vals) > 100000)
        self.assertTrue(len(lrows) > 100000)
        self.assertEqual(len(lrows), len(id_vals))

    #
    #   TEST _swap_instance_components(params: dict[str, any])
    #
    def test_swap_instance_components_known_system(self):
        """ Tests _swap_instance_components(well known params) -> correctly swaps/updates params """  
        params = {
            "rA_plus_rB": 0.306652,     "k": 0.703667,
            "J": 0.925529,              "qphot": 0.84,
            "ecosw": 0.005128,          "esinw": -0.011822,
            "inc": 86.381,              "bP": 0.355,
            "ecc": 0.012886,            "omega": 293.450947,
            "L3": 0,                    "phiS": 0.503265,
            "rA": 0.18000,              "rB": 0.12667,
            "LDA1": 0.6437,             "LDB1": 0.6445,
            "MA": 1.568,                "MB": 1.314,
        }

        swap_params = params.copy()
        _swap_instance_components(swap_params) # Will update swap_params in place

        # Unchanged - system as a whole
        self.assertEqual(swap_params["rA_plus_rB"], params["rA_plus_rB"])
        self.assertEqual(swap_params["inc"], params["inc"])
        self.assertEqual(swap_params["ecc"], params["ecc"])

        # Swapped component params
        self.assertEqual(swap_params["rA"], params["rB"])
        self.assertEqual(swap_params["rB"], params["rA"])
        self.assertEqual(swap_params["LDA1"], params["LDB1"])
        self.assertEqual(swap_params["LDB1"], params["LDA1"])
        self.assertEqual(swap_params["MA"], params["MB"])
        self.assertEqual(swap_params["MB"], params["MA"])

        # Updated ratios (invert the relations)
        self.assertEqual(swap_params["k"], 1 / params["k"])
        self.assertEqual(swap_params["J"], 1 / params["J"])
        self.assertEqual(swap_params["qphot"], 1 / params["qphot"])

        # Updated ascending node as the orgin is moved from star A to B : +pi
        self.assertEqual(swap_params["omega"], params["omega"] + 180 - 360)

        # Updated args of pariastron: ecc unchanged however omega is modified
        ecc, swap_omega = swap_params["ecc"], np.deg2rad(swap_params["omega"])
        self.assertAlmostEqual(swap_params["ecosw"], ecc * np.cos(swap_omega), 6)
        self.assertAlmostEqual(swap_params["esinw"], ecc * np.sin(swap_omega), 6)

        # bP recalculated, as it should now relate to what was originally rB
        exp_bp = impact_parameter(params["rB"], params["inc"],
                                  params["ecc"], -params["esinw"])
        self.assertAlmostEqual(swap_params["bP"], exp_bp, 6)


if __name__ == "__main__":
    unittest.main()
