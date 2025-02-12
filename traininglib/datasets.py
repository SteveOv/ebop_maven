"""
Functions for generating binary system instances and for writing/reading dataset files.
"""
# Use ucase variable names where they match the equivalent symbol and pylint can't find units alias
# pylint: disable=invalid-name, no-member, too-many-arguments, too-many-locals
from typing import Callable, Generator, List, Tuple, Iterable
import os
import errno
import sys
from pathlib import Path
from timeit import default_timer
from datetime import timedelta, datetime
import traceback
from multiprocessing import Pool
import hashlib

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

from deblib import orbital
from ebop_maven import deb_example

from traininglib import jktebop, param_sets


# Common options used when reading or writing a deb Example dataset file
ds_options = tf.io.TFRecordOptions(compression_type=None)


def make_dataset(instance_count: int,
                 file_count: int,
                 output_dir: Path,
                 generator_func: Callable[[str], Generator[dict[str, any], any, any]],
                 check_func: Callable[[dict[str, any]], bool],
                 valid_ratio: float=0.,
                 test_ratio: float=0,
                 file_prefix: str="dataset",
                 max_workers: int=1,
                 swap_if_deeper_secondary: bool=False,
                 save_param_csvs: bool=True,
                 verbose: bool=False,
                 simulate: bool=False) -> None:
    """
    This is a convenience wrapper for make_dataset_file() which handles parallel running by
    splitting a request into multiple files and calling make_dataset_file() for each within a pool.
    
    :instance_count: the number of training instances to create
    :file_count: the number of files to spread them over
    :output_dir: the directory to write the files to
    :generator_func: the function to call to infinitely generate systems, until closed,
    which must have arguments (file_stem: str) and return a Generator[dict[str, any]]
    :check_func: the boolean function to call to confirm an instance's params are usable.
    Will be called with the instance's **params dictionary (as kwargs)
    :valid_ratio: proportion of rows to be written to the validation files
    :test_ratio: proportion of rows to be written to the testing files
    :file_prefix: naming prefix for each of the dataset files
    :max_workers: maximum number of files to process concurrently
    :swap_if_deeper_secondary: whether the components are swapped if the secondary eclipse is deeper
    :save_param_csvs: whether to save csv files with full params in addition to the dataset files.
    :verbose: whether to print verbose progress/diagnostic messages
    :simulate: whether to simulate the process, skipping only file/directory actions
    """
    if valid_ratio + test_ratio > 1:
        raise ValueError("valid_ratio + test_ratio > 1")

    train_ratio = 1 - valid_ratio - test_ratio

    if verbose:
        print(f"""
Generate & write dEB system dataset for required number of instances
--------------------------------------------------------------------
The number of instances to generate:    {instance_count:,} across {file_count} file(s)
The dataset files to be written within: {output_dir}
The instance generator function is:     {generator_func.__name__}
The instance check function is:         {check_func.__name__}
Training : Validation : Test ratio is:  {train_ratio:.2f} : {valid_ratio:.2f} : {test_ratio:.2f}
File names will be prefixed with:       {file_prefix}
The maximum concurrent workers:         {max_workers}\n""")
        if swap_if_deeper_secondary:
            print("Instances' stars will be swapped where the secondary eclipse is the deeper")
        if simulate:
            print("Simulate requested so no files will be written.\n")
        elif save_param_csvs:
            print("Parameter CSV files will be saved to accompany dataset files.\n")

    if max_workers > 1 and _is_debugging():
        print("*** Detected a debugger so overriding the max_workers arg and setting it to 1 ***\n")
        max_workers = 1

    if verbose:
        print(f"Starting process at {datetime.now():%Y-%m-%d %H:%M:%S%z %Z}\n")
        start_time = default_timer()

    if not simulate:
        output_dir.mkdir(parents=True, exist_ok=True)

    # args for each make_dataset_file call as required by process_pool starmap
    file_inst_counts = list(_calculate_file_splits(instance_count, file_count))
    iter_params = (
        (count, file_ix, output_dir, generator_func, check_func, valid_ratio, test_ratio,
         file_prefix, swap_if_deeper_secondary, save_param_csvs, verbose, simulate)
            for (file_ix, count) in enumerate(file_inst_counts)
    )

    max_workers = min(file_count, max_workers or 1)
    if max_workers <= 1:
        # We could use a pool of 1, but it's useful to keep execution on the interactive proc
        for params in iter_params:
            make_dataset_file(*params)
    else:
        with Pool(max_workers) as pool:
            pool.starmap(make_dataset_file, iter_params)

    if verbose:
        print(f"\nFinished making a dataset of {file_count} file(s) within {output_dir}")
        print(f"The time taken was {timedelta(0, round(default_timer()-start_time))}.\n")


def make_dataset_file(inst_count: int,
                      file_ix: int,
                      output_dir: Path,
                      generator_func: Callable[[str], Generator[dict[str, any], any, any]],
                      check_func: Callable[[dict[str, any]], bool],
                      valid_ratio: float=0.,
                      test_ratio: float=0,
                      file_prefix: str="dataset",
                      swap_if_deeper_secondary: bool=False,
                      save_param_csvs: bool=True,
                      verbose: bool=False,
                      simulate: bool=False):
    """
    Will make a TensorFlow dataset (tfrecord) file and, optionally, an accompanying parameter csv
    file. The dataset file is actually split into up to three subfiles, one each in training,
    validation and testing subdirectories, with the ratios of instances sent to each dictated by
    the valid_ratio and test_ratio arguments (with train_ratio implied).

    Adds Gaussian noise to the LC/mags feature if a "snr" param is specified.
    
    :inst_count: the number of training instances to create
    :file_ix: the index number of this file. Is appended to file_prefix to make the file stem.
    :output_dir: the directory to write the files to
    :generator_func: the function to call to infinitely generate systems, until closed,
    which must have arguments (file_stem: str) and return a Generator[dict[str, any]]
    :check_func: the boolean function to call to confirm an instance's params are usable.
    Will be called with the instance's **params dictionary (as kwargs)
    :valid_ratio: proportion of rows to be written to the validation files
    :test_ratio: proportion of rows to be written to the testing files
    :file_prefix: naming prefix for each of the dataset files
    :swap_if_deeper_secondary: whether the components are swapped if the secondary eclipse is deeper
    :save_param_csvs: whether to save csv files with full params in addition to the dataset files.
    :verbose: whether to print verbose progress/diagnostic messages
    :simulate: whether to simulate the process, skipping only file/directory actions
    """
    # pylint: disable=too-many-branches, too-many-statements
    interp_kind = "cubic"
    file_stem = f"{file_prefix}{file_ix:03d}"

    # Don't use the built-in hash() function; it's not consistent across processes!!!
    seed = int.from_bytes(hashlib.shake_128(file_stem.encode("utf8")).digest(8))
    rng = np.random.default_rng(seed)

    ds_filename = f"{file_stem}.tfrecord"
    if save_param_csvs and not simulate:
        csv_file = output_dir / f"{file_stem}.csv"
        csv_file.unlink(missing_ok=True)
    else:
        csv_file = None

    inst_ix = 0
    generated_count, swap_count = 0, 0
    generator = generator_func(file_stem)
    ds_writers = [None] * 3
    ds_subsets = ["training", "validation", "testing"]
    try:
        # Set up the dataset files (deleting existing) and decide the distribution of instances
        inst_file_ixs = [1]*round(valid_ratio*inst_count) + [2]*round(test_ratio*inst_count)
        inst_file_ixs = [0]*(inst_count-len(inst_file_ixs)) + inst_file_ixs
        for subset_ix, ds_subset in enumerate(ds_subsets):
            if not simulate:
                ds_subset_file = output_dir / ds_subset / ds_filename
                if subset_ix in inst_file_ixs:
                    ds_subset_file.parent.mkdir(parents=True, exist_ok=True)
                    ds_writers[subset_ix] = tf.io.TFRecordWriter(f"{ds_subset_file}", ds_options)
                else:
                    ds_subset_file.unlink(missing_ok=True)
        rng.shuffle(inst_file_ixs)

        # This will continue to generate new instances until we have enough (==inst_count)
        while inst_ix < inst_count:
            inst_id = inst_ix
            try:
                params = next(generator)
                generated_count += 1
                inst_id = params.get("id", inst_id)
                params.setdefault("swapped", 0)

                # Extra params/features
                params.setdefault("dS_over_dP", orbital.ratio_of_eclipse_duration(params["esinw"]))
                phiS = params.setdefault("phiS", orbital.phase_of_secondary_eclipse(params["ecosw"],
                                                                                    params["ecc"]))

                is_usable = check_func(**params)
                if is_usable:
                    model_data = jktebop.generate_model_light_curve(file_prefix, **params)

                    # JKTEBOP will output NaNs for model LC if it doesn't like the params; skip
                    nans = np.isnan(model_data["delta_mag"])
                    if any(nans):
                        num = sum(nans)
                        print(f"{file_stem}[{inst_id}]: Dropping inst as jktebop generated",
                              f"{num} of {len(model_data)} mags as NaN in model LC.")
                        is_usable = False

                if is_usable:
                    # Find secondary eclipse & depth of both eclipses (assuming primary at 0).
                    # Look where the secondary is expected. Can't get argmax to work with islice
                    ixS = np.round(len(model_data) * phiS).astype(int)
                    ixS_range = slice(max(0, ixS-25), min(ixS+25, len(model_data)-1,))
                    ixS = ixS_range.start + np.argmax(model_data["delta_mag"][ixS_range])
                    depthS = params.setdefault("depthS", model_data[ixS]["delta_mag"])
                    depthP = params.setdefault("depthP", model_data[0]["delta_mag"])

                    # Give the caller the option to reject now we have more eclipse information
                    is_usable = check_func(**params)

                    # Optionally, swap the components & reposition the eclipses if secondary deeper.
                    if is_usable and swap_if_deeper_secondary and depthS > depthP:
                        _swap_instance_components(params)
                        is_usable = check_func(**params)
                        if is_usable:
                            # Regenerate the LC - assume it will work as params previously OK.
                            # A faster algo is to roll mags to 2ndary but this trains better.
                            model_data = jktebop.generate_model_light_curve(file_prefix, **params)
                            swap_count += 1
                            params["swapped"] = 1

                if is_usable:
                    # Optionally add Gaussian noise to the mags
                    noise_sigma = params.get("noise_sigma", None)
                    if noise_sigma:
                        # We apply the noise to fluxes, so revert delta mags to normalized flux
                        fluxes = np.power(10, np.divide(model_data["delta_mag"], -2.5))
                        noise = rng.normal(0., scale=noise_sigma, size=len(fluxes))
                        model_data["delta_mag"] = np.multiply(-2.5, np.log10(fluxes + noise))

                    # Optionally roll the phase folded mags based on the indicated phase shift
                    phase_shift = params.get("phase_shift", None)
                    if phase_shift:
                        shift = int(len(model_data) * phase_shift)
                        model_data["delta_mag"] = np.roll(model_data["delta_mag"], shift)

                    # Optionally add a y-shift up/down to offset the mags' zero point
                    mag_shift = params.get("mag_shift", None)
                    if mag_shift:
                        model_data["delta_mag"] += mag_shift

                    # We store mags_features for various supported bins values
                    mags_features = {}
                    interp = interp1d(model_data["phase"], model_data["delta_mag"], interp_kind)
                    for mag_name, mags_bins in deb_example.stored_mags_features.items():
                        # Ensure we don't waste a bin repeating the value for phase 0.0 & 1.0
                        new_phases = np.linspace(0., 1., mags_bins + 1)[:-1]
                        bin_model_data = np.array([new_phases, interp(new_phases)], dtype=np.double)
                        mags_features[mag_name] = bin_model_data[1]

                    # Write the appropriate dataset train/val/test file based on inst/file indices
                    # Use the params dict for labels & extra_features as it's now a superset of both
                    row = deb_example.serialize(inst_id, params, mags_features, params)
                    ds_writers[inst_file_ixs[inst_ix]].write(row)

                    if csv_file:
                        param_sets.write_to_csv(csv_file, [params], append=True)

                    inst_ix += 1
                    if verbose and inst_ix % 100 == 0:
                        print(f"{file_stem}: Generated {inst_ix:n} usable instances.")
            except StopIteration:
                break
            except Exception as exc: # pylint: disable=broad-exception-caught
                traceback.print_exception(exc)
                print(f"{file_stem}: instance {inst_ix} caused a {type(exc).__name__}: {exc}")

    finally:
        for ds_subset_file in ds_writers:
            if ds_subset_file and isinstance(ds_subset_file, tf.io.TFRecordWriter):
                ds_subset_file.close()
        generator.close()

    if verbose:
        if generated_count >= inst_count:
            for subset_ix, subset in enumerate(ds_subsets):
                saved_count = inst_file_ixs.count(subset_ix)
                if saved_count:
                    print(f"{file_stem}:", "Simulated saving" if simulate else "Saved",
                        f"{saved_count:n} {subset} instance(s) to",
                        f"{output_dir.name}/{subset}/{ds_filename}")
        else:
            print(f"{file_stem}: \033[93m\033[1m!!!Failed after generating",
                  f"{generated_count-1} of {inst_count} instances!!!\033[0m")
        if swap_count > 0:
            print(f"{file_stem}: Swapped the components of {swap_count} instance(s)",
                  "where the secondary eclipse was deeper.")


def create_map_func(mags_bins: int = deb_example.default_mags_bins,
                    mags_wrap_phase: float = deb_example.default_mags_wrap_phase,
                    ext_features: List[str] = None,
                    labels: List[str] = None,
                    augmentation_callback: Callable[[tf.Tensor], tf.Tensor] = None,
                    scale_labels: bool=True,
                    include_id: bool=False) -> Callable:
    """
    Configures and returns a dataset map function for deb_examples. The map function is used by
    TFRecordDataset().map() to deserialize each raw tfrecord row into the corresponding features
    and labels required for model training or testing.
        
    In addition to deserializing the rows, the map function supports the option of augmenting the
    mags_feature data by supplying a reference to an augmentation_callback function. This callback
    will be called from a graphed function so must be compatible with tf.function.
    A simple example to augment the mags_feature with additive Gaussian noise:
    ```Python
    @tf.function
    def sample_aug_callback(mags_feature: tf.Tensor) -> tf.Tensor:
        return mags_feature + tf.random.normal(mags_feature.shape, stddev=0.005)
    ```

    The mags_wrap_phase argument controls at which point the cyclic, phase normalized mags data is
    wrapped when read. This can be in the range [0, 1] or None (in which case the mags data is
    adaptively wrapped to centre on the mid-point between the primary and secondary eclipses).

    :mags_bins: the width of the mags to publish
    :mags_wrap_phase: the wrap phase of the mags to publish, or None to use adaptive wrap
    :ext_features: chosen subset of available ext_features, in the requested order, or all if None
    :labels: chosen subset of available labels, in requested order, or all if None
    :augmentation_callback: optional function with which client code can augment mags_feature data.
    :scale_labels: whether to apply scaling to the map function's returned labels
    :include_id: whether to include the id in the function's returned tuple. If true, it returns
    (id, (mags_feature, ext_features), labels) per inst else ((mags_feature, ext_features), labels)
    :returns: the newly generated map function
    """
    # pylint: disable=too-many-arguments
    mags_key = deb_example.create_mags_key(mags_bins)
    if mags_wrap_phase is not None:
        mags_wrap_phase %= 1 # Treat the wrap as cyclic & force it into the range [0, 1)

    if ext_features is not None:
        chosen_ext_feat_and_defs = \
            { ef: deb_example.extra_features_and_defaults[ef] for ef in ext_features}
    else:
        chosen_ext_feat_and_defs = deb_example.extra_features_and_defaults

    if labels is not None:
        chosen_lab_and_scl = { l: deb_example.labels_and_scales[l] for l in labels }
    else:
        chosen_lab_and_scl = deb_example.labels_and_scales

    # Define the map function with the two, optional perturbing actions on the mags feature
    def map_func(record_bytes):
        example = tf.io.parse_single_example(record_bytes, deb_example.description)

        # Get mags feature and reshape to match ML model's requirements; from (#bins,) to (#bins, 1)
        # Assume mags will have been stored with phase implied by index and primary eclipse at zero
        mags_feature = tf.reshape(example[mags_key], shape=(mags_bins, 1))

        if mags_wrap_phase is not None:
            roll_phase = mags_wrap_phase
        else:
            # Adaptive; chosen to centre mags on the midpoint between primary & secondary eclipses
            # With the primary initially at phase 0, the midpoint is half the secondary phase
            roll_phase = 0.5 + example.get("phiS", 0.5) / 2

        # Now roll the mags to match the requested wrap phase. For example, if the roll phase
        # is 0.75 then the mags will be rolled right by 0.25 phase so that those mags originally
        # beyond phase 0.75 are rolled round to lie before phase 0 (effectively phases -0.25 to 0).
        # Combine with any roll augmentation so we only incur the overhead of rolling once
        roll_shift = 0 if roll_phase == 0 else int(mags_bins * (1.0 - roll_phase))
        if roll_shift != 0:
            if roll_shift > mags_bins // 2:
                roll_shift -= mags_bins
            mags_feature = tf.roll(mags_feature, [roll_shift], axis=[0])

        # Augmentations: not all potential augmentations have in-place updates (i.e. tf.roll) so we
        # endure the overhead of send/return rather than using "byref" behaviour of a mutable arg.
        if augmentation_callback:
            mags_feature = augmentation_callback(mags_feature)

        # The Extra features: ignore unknown fields and use default if not found
        ext_features = [example.get(k, d) for (k, d) in chosen_ext_feat_and_defs.items()]
        ext_features = tf.reshape(ext_features, shape=(len(ext_features), 1))

        # Copy labels in the expected order & optionally apply any scaling
        if scale_labels:
            labels = [example[k] * s for k, s in chosen_lab_and_scl.items()]
        else:
            labels = [example[k] for k in chosen_lab_and_scl]

        if include_id:
            return (example["id"], (mags_feature, ext_features), labels)
        return ((mags_feature, ext_features), labels)
    return map_func


def create_dataset_pipeline(dataset_files: Iterable[str],
                            batch_size: float=100,
                            map_func: Callable=create_map_func(),
                            filter_func: Callable=None,
                            shuffle: bool=False,
                            reshuffle_each_iteration: bool=False,
                            max_buffer_size: int=1000000,
                            prefetch: int=tf.data.AUTOTUNE,
                            seed: int=42) -> Tuple[tf.data.TFRecordDataset, int]:
    """
    Creates the requested TFRecordDataset pipeline.

    The filter_func must be in a form simlar to the example below, where;
    - it will be called from a graphed function so must be compatible with tf.function
    - id argument is each instance's id of type tf.Tensor(shape=(), dtype=string)
    - x argument is the instance's features as ( mags_feature, extra_features ), where
        - mags_feature is of type tf.Tensor(shape=(#bins,1), dtype=float)
        - extra_features is of type tf.Tensor(shape=(#extra_features, 1), dtype=float) 
    - y argument is the instance's labels as tf.Tensor(shape=(#labels, 1), dtype=float)
    - returns a boolean

    ```python
    @tf.function
    def sample_filter_func(id, x, y):
        ix_phiS = [*deb_example.extra_features_and_defaults.keys()].index("phiS")
        return x[1][ix_phiS][0] > 0.5
    ```
    or
    ```python
    @tf.function
    def sample_filter_func(id, x, y):
        return id == "CW Eri"
    ```

    The filter_func is positioned after map_func in the pipeline so can only
    filter on mapped values (& cannot see any that are omitted).
    
    :dataset_files: the source tfrecord dataset files.
    :batch_size: the relative size of each batch. May be set to 0 (no batch, effectively all rows),
    <1 (this fraction of all rows) or >=1 this size (will be rounded)
    :map_func: the map function to use to deserialize each row
    :filter_func: an optional func to filter the results (must be a tf.function)
    :shuffle: whether to include a shuffle step in the pipeline
    :reshuffle_each_iteration: whether the shuffle step suffles on each epoch
    :max_buffer_size: the maximum size of the shuffle buffer
    :prefetch: the number of prefetch operations to perform, or leave to autotune
    :seed: seed for any random behaviour
    :returns: a tuple of (dataset pipeline, row count). The row count is the total
    rows without any optional filtering applied.
    """
    # pylint: disable=too-many-arguments
    # Explicitly check the dataset_files otherwise we may get a cryptic errors further down.
    if dataset_files is None or len(dataset_files) == 0:
        raise ValueError("No dataset_files specified")
    for file in dataset_files:
        if not Path(file).exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    # Read through once to get the total number of records
    ds = tf.data.TFRecordDataset(list(dataset_files), num_parallel_reads=100,
                                 compression_type=ds_options.compression_type)
    row_count = ds.reduce(0, lambda count, _: count+1).numpy()

    # Now build the full pipeline
    if shuffle or reshuffle_each_iteration:
        buffer_size = min(row_count, max_buffer_size)
        ds = ds.shuffle(buffer_size, seed, reshuffle_each_iteration=reshuffle_each_iteration)

    ds = ds.map(map_func)

    if filter_func:
        ds = ds.filter(filter_func)

    if batch_size:
        if batch_size < 1:
            batch_size = int(np.ceil(row_count * batch_size))
        else:
            batch_size = min(row_count, int(batch_size))
        if batch_size:
            ds = ds.batch(batch_size, drop_remainder=True)

    if prefetch:
        ds = ds.prefetch(prefetch)
    return (ds, row_count)


def iterate_dataset(dataset_files: Iterable[str],
                    mags_bins: int = deb_example.default_mags_bins,
                    mags_wrap_phase: float = deb_example.default_mags_wrap_phase,
                    ext_features: List[str] = None,
                    labels: List[str]=None,
                    identifiers: List[str]=None,
                    scale_labels: bool=False,
                    filter_func: Callable=None,
                    augmentation_callback: Callable[[tf.Tensor], tf.Tensor] = None,
                    max_instances: int = np.inf):
    """
    Utility/diagnostics function which will parse a saved dataset yielding rows,
    and within the rows labels and features, which match the requested criteria.
    
    The rows are yielded in the order in which they appear in the supplied dataset files.
    
    For details of filter_func argument see documentation for create_dataset_pipeline()

    For details of augmentation_callback argument see documentation for create_map_func()

    This function is not for use when training a model; for that, requirement use
    create_dataset_pipeline() directly. Instead this gives easy access to the
    contents of a dataset for diagnostics lookup, testing or plotting.

    :dataset_files: the set of dataset files to parse
    :mags_bins: the width of the mags to publish
    :mags_wrap_phase: the wrap phase of the mags to publish
    :ext_features: a chosen subset of the available features, in this order, or all if None
    :labels: a chosen subset of the available labels, in this order, or all if None
    :identifiers: optional list of ids to yield, or all ids if None
    :filter_func: an optional func to filter the results (must be a tf.function)
    :augmentation_callback: optional function with which client code can augment mags_feature data.
    :max_instances: the maximum number of instances to yield
    :returns: for each matching row yields a tuple of (id, mags vals, ext feature vals, label vals)
    """
    # pylint: disable=too-many-arguments, too-many-locals
    if identifiers is not None and len(identifiers) < max_instances:
        max_instances = len(identifiers)

    map_func = create_map_func(mags_bins, mags_wrap_phase, ext_features, labels,
                               augmentation_callback, scale_labels, include_id=True)
    (ds, _) = create_dataset_pipeline(dataset_files, 0, map_func, filter_func)

    yield_count = 0
    for id_val, (mags_val, feat_vals), lab_vals in ds.as_numpy_iterator():
        id_val = id_val.decode(encoding="utf8")
        if identifiers is None or id_val in identifiers:
            # Primarily, create_map_func supports a pipeline for training an ML model where the mags
            # and ext_features for each inst is required to be shaped as (#bins, 1) and (#feats, 1).
            # Here our client code expects shapes of (#bins,) and (#feats,).
            yield id_val, mags_val[:, 0], feat_vals[:, 0], lab_vals
            yield_count += 1
            if yield_count >= max_instances:
                break

def read_dataset(dataset_files: Iterable[str],
                 mags_bins: int = deb_example.default_mags_bins,
                 mags_wrap_phase: float = deb_example.default_mags_wrap_phase,
                 ext_features: List[str] = None,
                 labels: List[str]=None,
                 identifiers: List[str]=None,
                 scale_labels: bool=False,
                 filter_func: Callable=None,
                 augmentation_callback: Callable[[tf.Tensor], tf.Tensor] = None,
                 max_instances: int = np.inf) \
            -> Tuple[np.ndarray, np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Wrapper around iterate_dataset() which handles the iteration and returns separate
    np ndarrays for the dataset ids, mags values, feature values and label values.
    
    For details of filter_func argument see documentation for create_dataset_pipeline()

    For details of augmentation_callback argument see documentation for create_map_func()

    The labels array is structured, so labels may be accessed with their names, for example;
    ```Python
    ecosw = labels[0]["ecosw"]
    ```
    
    This may not be hugely performant, especially with large datasets, but it's for convenience.

    :dataset_files: the set of dataset files to parse
    :mags_bins: the width of the mags to publish
    :mags_wrap_phase: the wrap phase of the mags to publish
    :ext_features: a chosen subset of the available features, in this order, or all if None
    :labels: a chosen subset of the available labels, in this order, or all if None
    :identifiers: optional list of ids to yield, or all ids if None
    :scale_values: if True values will be scaled
    :filter_func: an optional func to filter the results (must be a tf.function)
    :augmentation_callback: optional function with which client code can augment mags_feature data.
    :max_instances: the maximum number of instances to return
    :returns: a Tuple[NDArray[#insts, 1], NDArray[#insts, #bins], NDArray[#insts, #feats],
    NDArray[#insts, #labels]], with the labels being a structured NDArray supporting named columns
    """
    # pylint: disable=too-many-arguments, too-many-locals
    if labels is not None:
        labels = [l for l in labels if l in deb_example.labels_and_scales]
    else:
        labels = list(deb_example.labels_and_scales.keys())

    ids, mags_vals, feature_vals, label_vals = [], [], [], []
    for row in iterate_dataset(dataset_files, mags_bins, mags_wrap_phase,
                               ext_features, labels, identifiers, scale_labels,
                               filter_func, augmentation_callback, max_instances):
        ids += [row[0]]
        mags_vals += [row[1]]
        feature_vals += [row[2]]
        label_vals += [tuple(row[3])]

    # Need to sort the data in the order of the requested ids (if given).
    # Not hugely performant, but we only ever expect short lists of indices.
    if identifiers is not None and len(identifiers) > 0:
        indices = [ids.index(i) for i in identifiers if i in ids]
        ids = [ids[ix] for ix in indices]
        mags_vals = [mags_vals[ix] for ix in indices]
        feature_vals = [feature_vals[ix] for ix in indices]
        label_vals = [label_vals[ix] for ix in indices]

    # Turn label vals into a structured array
    dtype = [(name, np.dtype(float)) for name in labels]
    return np.array(ids), np.array(mags_vals), np.array(feature_vals), np.array(label_vals, dtype)


def _calculate_file_splits(instance_count: int, file_count: int) -> list[int]:
    """
    Calculates the most equal split of instance (sub) counts across the
    requested number of files. The split is as even as possible with the
    last file given any shortfall. If file_count is 1 it gets everything.
    
    :instance_count: the number of instances
    :file_count: the number of files
    :returns: a list of file instance counts of length file_count
    """
    file_instance_counts = [int(np.ceil(instance_count / file_count))] * (file_count-1)
    file_instance_counts += [instance_count - sum(file_instance_counts)]
    return file_instance_counts


def _is_debugging() -> bool:
    """
    Whether the code thinks we are running under the watchful eye of a debugger
    """
    # The default way, but no longer appears to be working in later versions of VSCode
    gt = getattr(sys, "gettrace", None)
    debugging = gt is not None and gt() is not None
    if not debugging and hasattr(sys, "monitoring"):
        # This may work in new version of python - works for me in VSCode & python 3.12
        debugging = sys.monitoring.get_tool(sys.monitoring.DEBUGGER_ID) is not None
    return debugging


def _swap_instance_components(params: dict[str, any]):
    """
    Handle swapping the A and B stars, and its effect on the various instance parameters.

    :params: the params dict (passed byref) which is updated in place
    """
    # The "system as a whole" params such as rA_plus_rB, e and L3 are unchanged.
    # However, we need to flip the ratios as the A and B stars are swapped.
    params["k"] = np.reciprocal(params["k"])
    params["J"] = np.reciprocal(params["J"])
    params["qphot"] = np.reciprocal(params["qphot"])
    if "LB_over_LA" in params:
        params["LB_over_LA"] = np.reciprocal(params["LB_over_LA"])

    # We update omega to omega+pi; the ascending node is unchanged, but moving origin to star B
    # changes the argument of periastron from ω to ω+pi. Using the relations sin(ω+pi)=-sin(ω)
    # & cos(ω+pi)=-cos(ω) and with e unchanged, we just need to negate the Poincare elements.
    params["ecosw"] *= -1
    params["esinw"] *= -1
    if "omega" in params:
        params["omega"] = (params["omega"] + 180) % 360

    # Eccentricity will be unchanged but we need it for the following updates
    e = params["ecc"] if "ecc" in params else np.sqrt(params["ecosw"]**2 + params["esinw"]**2)

    # These are related to the Poincare elements.
    if "phiS" in params:
        params["phiS"] = orbital.phase_of_secondary_eclipse(params["ecosw"], e)
    if "dS_over_dP" in params:
        params["dS_over_dP"] = np.reciprocal(params["dS_over_dP"])

    # Swap over the limb darkening parameters and any (optional) star specific informational params
    for pattern in ["LD{0}", "LD{0}1", "LD{0}2", "r{0}", "R{0}", "M{0}", "L{0}", "Teff{0}"]:
        keyA = pattern.format("A")
        keyB = pattern.format("B")
        if keyA in params and keyB in params:
            params[keyA], params[keyB] = params[keyB], params[keyA]

    # Recalculate the impact parameters as we have changed the primary star. It's not just a case of
    # swapping the values, as both impact params are related to the primary's fractional radius.
    params["bP"] = orbital.impact_parameter(params["rA"], params["inc"], e, params["esinw"], False)
    params["bS"] = orbital.impact_parameter(params["rA"], params["inc"], e, params["esinw"], True)
