"""
Functions for building TensorFlow datasets.
"""
# Use ucase variable names where they match the equivalent symbol and pylint can't find units alias
# pylint: disable=invalid-name, no-member
from typing import Iterator, Dict, List, Union
from pathlib import Path
import random
import traceback
from timeit import default_timer
from datetime import timedelta
from multiprocessing import Pool
import json
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.interpolate import interp1d
import astropy.units as u
from lightkurve import LightCurve
import tensorflow as tf

from .libs import param_sets, jktebop, orbital, lightcurve, deb_example

# Common options used when writing a dataset file
_ds_options = tf.io.TFRecordOptions(compression_type=None)

def make_dataset_files(trainset_files: Iterator[Path],
                       output_dir: Path,
                       valid_ratio: float=0.,
                       test_ratio: float=0,
                       interp_kind: str="cubic",
                       resume: bool=False,
                       max_workers: int=1,
                       seed: float=42,
                       verbose: bool=True,
                       simulate: bool=True):
    """
    Will make the an equivalent set of TensorFlow dataset (tfrecord) files from
    the input trainset csv files. Up to three dataset files, one each for
    training, validation and testing, will be created for each input file in
    equivalently named subdirectories of the output_dir argument. The files'
    contents are selected randomly from the input in the proportions indicated
    by the valid_ratio and test_ratio arguments. Each output row will include
    a freshly generated phase folded and model lightcurve & additional features,
    and the training labels taken from the input.

    :trainset_files: the input trainset csv files
    :output_dir: the parent directory for the dataset files, if None the same as the input
    :valid_ratio: proportion of rows to be written to the validation files
    :test_ratio: proportion of rows to be written to the testing files
    :interp_kind: the kind of interpolation used to reduce the the lightcurve models
    :resume: whether we are to attempt to resume from a previous "make"
    :max_workers: maximum number of files to process concurrently
    :seed: the seed ensures random selection of subsets are repeatable
    :verbose: whether to print verbose progress/diagnostic messages
    :simulate: whether to simulate the process, skipping only file/directory actions
    """
    start_time = default_timer()
    trainset_files = sorted(trainset_files)
    file_count = len(trainset_files)
    train_ratio = 1 - valid_ratio - test_ratio
    if verbose:
        print(f"""
Build training datasets from testset csvs.
------------------------------------------
The number of input trainset files is:  {file_count}
Output dataset directory:               {output_dir}
Resume previous job is set to:          {'on' if resume else 'off'}
Training : Validation : Test ratio is:  {train_ratio:.2f} : {valid_ratio:.2f} : {test_ratio:.2f}
The model interpolation kind is:        {interp_kind}
The maximum concurrent workers:         {max_workers}
The random seed to use for selections:  {seed}\n""")
        if simulate:
            print("Simulate requested so no files will be written.\n")

    # args for each make_dataset_file call as required by process_pool starmap
    iter_params = (
    (f, output_dir, valid_ratio, test_ratio, interp_kind, resume, seed, verbose, simulate)
        for f in trainset_files
    )

    max_workers = min(file_count, max_workers or 1)
    if max_workers <= 1:
        # We could use a pool of 1, but keep execution on the interactive proc
        for params in iter_params:
            make_dataset_file(*params)
    else:
        with Pool(max_workers) as pool:
            pool.starmap(make_dataset_file, iter_params)

    print(f"\nFinished making the dataset for {file_count} trainset file(s) to {output_dir}")
    print(f"The time taken was {timedelta(0, round(default_timer()-start_time))}.")


def make_dataset_file(trainset_file: Path,
                      output_dir: Path=None,
                      valid_ratio: float=0.,
                      test_ratio: float=0.,
                      interp_kind: str="cubic",
                      resume: bool=False,
                      seed: float=42,
                      verbose: bool=True,
                      simulate: bool=False) -> None:
    """
    Will make the an equivalent set of TensorFlow dataset (tfrecord) files from
    the input trainset csv file. Up to three output files, one each for
    training, validation and testing, will be created in equivalently named
    subdirectories of the output_dir argument. The contents of the three files
    will be selected randomly from the input in the proportions indicated
    by the valid_ratio and test_ratio arguments. Each output row will include
    a freshly generated phase folded and model lightcurve & additional features,
    and the training labels taken from the input.

    :trainset_file: the input trainset csv file
    :output_dir: the parent directory for the dataset files, if None the same as the input
    :valid_ratio: proportion of rows to be written to the validation file
    :test_ratio: proportion of rows to be written to the testing file
    :interp_kind: the kind of interpolation used to sample the the lightcurve model
    :resume: whether we are to attempt to resume from a previous "make"
    :seed: the seed ensures random selection of subsets are repeatable
    :verbose: whether to print verbose progress/diagnostic messages
    :simulate: whether to simulate the process, skipping only file/directory actions
    """
    label = trainset_file.stem
    output_dir = trainset_file.parent if output_dir is None else output_dir
    this_seed = f"{trainset_file.name}/{seed}"

    # Work out the subsets & files now, before generating, to see if we can skip on resume
    subsets = ["training", "validation", "testing"]
    subset_values = [(1 - valid_ratio - test_ratio), valid_ratio, test_ratio]
    subset_files = [output_dir / subset / f"{trainset_file.stem}.tfrecord" for subset in subsets]
    if resume and all(subset_files[i].is_file() == (subset_values[i] > 0) for i in range(3)):
        if verbose:
            print(f"{label}: Resume is on and all expected output files exist. Skipping.")
        return

    # Current approach builds the full set of outputs in memory, shuffles an index before selecting
    # contiguous index subset blocks & saving the indexed data rows to files. A more efficient algo
    # would be to create & shuffle the index and open the subset output files in advance, then write
    # the rows as we go. However, for that to work we need to know the total row count in advance.
    rows = []
    for counter, params in enumerate(param_sets.read_param_sets_from_csv(trainset_file), 1):
        params.setdefault("gravA", 0.)
        params.setdefault("gravB", 0.)
        params.setdefault("reflA", -100)
        params.setdefault("reflB", -100)

        try:
            # model_data's shape is (2, rows) with phase in [0, :] and mags in [1, :]
            model_data = jktebop.generate_model_light_curve("trainset_", **params)
            if np.isnan(np.min(model_data[1])):
                # Checking for a Heisenbug where a model is somehow assigned NaN for at least 1 mag
                # value, subsequently causing training to fail. Adding mitigation/error reporting
                # seems to stop it happening despite the same source params, args and seed being
                # used. I'll leave this in place to report if it ever happens again. I think the
                # issue was caused by passing "bad" params to JKTEBOP which is why it's not repeated
                print(f"{label}[{params['id']}]: Replacing NaN/Inf in processed LC.")
                np.nan_to_num(x=model_data[1], copy=False)

            # We apply and store various supported configs of bins and wrap phase
            interpolator = interp1d(model_data[0], model_data[1], kind=interp_kind)
            mags_features = {}
            for mag_name, (mags_bins, wrap_phase) in deb_example.stored_mags_features.items():
                # Ensure we don't waste a row on a phase 1.0 value already covered by phase 0.0
                new_phases = np.linspace(0., 1., mags_bins + 1)[:-1]
                bin_model_data = np.array([new_phases, interpolator(new_phases)], dtype=np.double)
                if wrap_phase and 0 < wrap_phase < 1:
                    bin_model_data[0, bin_model_data[0] > wrap_phase] -= 1.
                    shift = bin_model_data.shape[1] - np.argmin(bin_model_data[0])
                    bin_model_data = np.roll(bin_model_data, shift, axis=1)
                mags_features[mag_name] = bin_model_data[1]

            # These are the extra features used for predictions alongside the LC.
            extra_features = {
                "phiS": lightcurve.expected_secondary_phase(params["ecosw"], params["ecc"]),
                "dS_over_dP": lightcurve.expected_ratio_of_eclipse_duration(params["esinw"]),
            }

            rows.append(deb_example.serialize(params["id"], params, mags_features, extra_features))

            if verbose and counter % 100 == 0:
                print(f"{label}: Processed {counter} instances.")

        except Exception as exc: # pylint: disable=broad-exception-caught
            traceback.print_exc(exc)
            print(f"{label}: Skipping instance {counter} which caused exc: {params}")

    rows_total = len(rows)
    if verbose:
        print(f"{label}: Finished processing {rows_total} instances.")

    # Turn the subset ratios into counts now we know total number of rows
    subset_values[2] = round(subset_values[2] * rows_total)
    subset_values[1] = round(subset_values[1] * rows_total)
    subset_values[0] = rows_total - sum(subset_values[1:]) # ensure training has all that's left

    # Set up a shuffled index. We use an index rather than shuffling the rows
    # directly, which is just as quick, so we can undo the suffle when saving.
    row_indices = np.arange(rows_total)
    random.Random(this_seed).shuffle(row_indices)
    subset_slice_start = 0
    for subset, subset_file, subset_count in zip(subsets, subset_files, subset_values):
        short_name = f"{subset_file.parent.name}/{subset_file.name}"
        if simulate:
            msg = f"{label}: Simulated saving {subset_count} {subset} instance(s) to {short_name}"
        elif subset_count > 0:
            # (Over)write the subset file. Rows are selected as a contiguous block of the shuffled
            # indices. which are then re-sorted so the rows is written in the original order.
            subset_slice = slice(subset_slice_start, subset_slice_start + subset_count)
            subset_file.parent.mkdir(parents=True, exist_ok=True)
            with tf.io.TFRecordWriter(f"{subset_file}", _ds_options) as ds:
                for sorted_ix in sorted(row_indices[subset_slice]):
                    ds.write(rows[sorted_ix])
            subset_slice_start = subset_slice.stop
            msg = f"{label}: Saved {subset_count} {subset} instance(s) to {short_name}"
        else:
            # Delete the existing file which may be left from previous run or we
            # will be left with too many rows, and duplicates, over the subsets.
            subset_file.unlink(missing_ok=True)
        if verbose:
            print(msg)


def make_formal_test_dataset(config_file: Path,
                             output_dir: Path,
                             fits_cache_dir: Path,
                             target_names: Iterator[str]=None,
                             verbose: bool=True,
                             simulate: bool=True) -> Path:
    """
    This creates a dataset based on real systems; their TESS light curve data with derived features
    & labels from published works. The information required to carry this out is supplied as a json
    file in the input_file argument. The following example shows one target from the input file and
    the tags that are used in this function. Not all tags are mandatory; mission defaults to "TESS",
    author to "SPOC", exptime to "short", "quality_bitmask" to "default", "flux_column" to
    "sap_flux", ecc and omega are assumed to be zero if omitted. bP will be calculated from other
    values if omitted.

    Example config:
    {
        "V436 Per": {
            "mission": "TESS" | "HLPSP",
            "author": "SPOC" | "TESS-SPOC",
            "exptime": "long" | "short" | "fast" | int (s),
            "sectors": {
                "18": {
                    "quality_bitmask": "hardest" | "hard" | "default",
                    "flux_column": "sap_flux" | "pdcsap_flux",
                    "primary_epoch": 1813.201149,
                    "period": 25.935953,
                    "ecc": 0.3835,
                    "omega": 109.56,
                    "labels": {
                        "rA_plus_rB": 0.08015,
                        "k": 1.097,
                        "bP": 0.59,
                        "inc": 87.951,
                        "ecosw": -0.12838,
                        "esinw": 0.3614,
                        "J": 1.041,
                        "L3": -0.003
                    }
                }
            }
        }
    }

    :input_file: the input json file containing the parameters for one or more targets
    :output_dir: the directory to write the output dataset tfrecord file
    :fits_cache_dir: the parent directory under which to cache downloaded fits files
    :target_names: a list of targets to select from input_file, or None for all
    :verbose: whether to print verbose progress/diagnostic messages
    :simulate: whether to simulate the process, skipping only file/directory actions
    :returns: the Path of the newly created dataset file
    """
    # pylint: disable=invalid-name
    start_time = default_timer()
    fits_cache_dir = fits_cache_dir if fits_cache_dir else output_dir
    smooth_fit = False

    if verbose:
        print(f"""
Build formal test dataset based on downloaded lightcurves from TESS targets.
----------------------------------------------------------------------------
The input configuration files is:   {config_file}
Output dataset will be written to:  {output_dir}
Downloaded fits are cached in:      {fits_cache_dir}
Selected targets are:               {', '.join(target_names) if target_names else 'all'}\n""")
        if simulate:
            print("Simulate requested so no dataset will be written, however fits are cached.\n")

    with open(config_file, mode="r", encoding="utf8") as f:
        targets = json.load(f)
        if target_names:
            targets = { name: targets[name] for name in target_names if name in targets }

    out_file = output_dir / f"{config_file.stem}.tfrecord"
    if not simulate:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        ds = tf.io.TFRecordWriter(f"{out_file}", _ds_options)

    try:
        inst_counter = 0
        for target_counter, (target, target_cfg) in enumerate(targets.items(), start=1):
            if verbose:
                print(f"\nProcessing target {target_counter} of {len(targets)}: {target}")

            fits_dir = fits_cache_dir / re.sub(r'[^\w\d-]', '_', target.lower())
            for sector in [int(s) for s in target_cfg["sectors"].keys() if s.isdigit()]:
                inst_id = f"{target}/{sector:0>4}"
                sector_cfg = sector_config_from_target(sector, target_cfg)

                # These are mandatory, so error if missing
                labels = sector_cfg["labels"]
                period = sector_cfg["period"] * u.d
                pe = sector_cfg["primary_epoch"]

                # Get the Lightcurve that matches the target & sector criteria and preprocess it.
                # Will attempt to service the request from previously downloaded fits if present.
                lc = prepare_lc_for_target_sector(target, sector, sector_cfg, fits_dir, verbose)
                pe = lightcurve.to_lc_time(pe, lc)

                # Produce multiple mags set (varying #bins & wrap phase) available for serialization
                if verbose:
                    print(f"{inst_id}: Creating phase normalized, folded lightcurves about",
                          f"{pe.format} {pe} & {period}.")
                mags_features = {}
                for mag_name, (mags_bins, wrap_phase) in deb_example.stored_mags_features.items():
                    # Phase folding the light-curve, then interpolate for the mags features
                    wrap_phase = u.Quantity(wrap_phase)
                    fold_lc = lc.fold(period, pe, wrap_phase=wrap_phase, normalize_phase=True)
                    _, mags = lightcurve.get_reduced_folded_lc(fold_lc, mags_bins, wrap_phase,
                                                                smooth_fit=smooth_fit)
                    mags_features[mag_name] = mags

                # omega & ecc are not used as labels but we need them for phiS and impact params
                ecosw, esinw = labels["ecosw"], labels["esinw"]
                omega = sector_cfg.get("omega", None) \
                    or np.rad2deg(np.arctan(np.divide(esinw, ecosw))) if ecosw else 0
                ecc = sector_cfg.get("ecc", None) \
                    or np.divide(ecosw, np.cos(np.deg2rad(omega))) if ecosw else 0

                # May need to calculate sini and cosi if not present
                inc_rad = np.deg2rad(labels["inc"])
                labels.setdefault("sini", np.sin(inc_rad))
                labels.setdefault("cosi", np.cos(inc_rad))

                # May need to calculate the primary impact parameter label as it's rarely published.
                bP = labels.get("bP", None)
                if bP is None:
                    rA = np.divide(labels["rA_plus_rB"], np.add(1, labels["k"]))
                    labels["bP"] = orbital.impact_parameter(rA, labels["inc"] * u.deg, ecc, None,
                                                            esinw, orbital.EclipseType.PRIMARY)
                    if verbose:
                        print(f"{inst_id}: No impact parameter (bP) supplied;",
                                f"calculated rA = {rA} and then bP = {labels['bP']}")

                # Now assemble the extra features needed: phiS (phase secondary) and dS_over_dP
                extra_features = {
                    "phiS": lightcurve.expected_secondary_phase(labels["ecosw"], ecc),
                    "dS_over_dP": lightcurve.expected_ratio_of_eclipse_duration(labels["esinw"])
                }

                # Serialize the labels, folded mags (lc) & extra_features as a deb_example and write
                if not simulate:
                    if verbose:
                        print(f"{inst_id}: Saving serialized instance to dataset:", out_file)
                    ds.write(deb_example.serialize(inst_id, labels, mags_features, extra_features))
                elif verbose:
                    print(f"{inst_id}: Simulated saving serialized instance to dataset:", out_file)
                inst_counter += 1
    finally:
        if ds:
            ds.close()

    action = "Finished " + ("simulating the saving of" if simulate else "saving")
    print(f"\n{action} {inst_counter} instance(s) from {len(targets)} target(s) to", out_file)
    print(f"The time taken was {timedelta(0, round(default_timer()-start_time))}.")
    return out_file


def inspect_dataset(dataset_files: Union[Path, Iterator[Path]],
                    identifiers: List[str]=None,
                    scale_labels: bool=False):
    """
    Utility/diagnostics function which will parse a saved dataset yielding each
    row that matches the passed list of ids (or every row if ids is None). The
    rows are yielded in the order in which they appear in the sorted tfrecords.

    :dataset_files: the set of dataset files to parse
    :identifiers: optional list of ids to yield, or all ids if None
    :scale_values: if True values will be scaled 
    """

    if isinstance(dataset_files, Path):
        dataset_files = [f"{dataset_files.resolve()}"]
    else:
        dataset_files = [f"{f.resolve()}" for f in sorted(dataset_files)]

    label_names_and_scales = deb_example.labels_and_scales
    for raw_record in tf.data.TFRecordDataset(dataset_files, _ds_options.compression_type):
        # We know the id is encoded as a utf8 str in a bytes feature
        example = tf.io.parse_single_example(raw_record, deb_example.description)
        identifier = example["id"].numpy().decode(encoding="utf8")
        if not identifiers or identifier in identifiers:
            if scale_labels:
                labels = { k: example[k].numpy()*sc for k, sc in label_names_and_scales.items() }
            else:
                labels = { k: example[k].numpy() for k in label_names_and_scales }
            mags = { k: example[k].numpy() for k in deb_example.stored_mags_features }
            ext_features = { k: example[k].numpy() for k in deb_example.extra_features_and_defaults}
            yield (identifier, labels, mags, ext_features)


def plot_dataset_instance_model_feature(dataset_files: Union[Path, Iterator[Path]],
                                        identifier: str,
                                        output: Union[Path, Axes],
                                        wrap_phase: float=0.75):
    """
    Utility function to produce a plot of the requested dataset instance's model feature

    :dataset_files: the set of dataset files to parse
    :identifier: the identifier of the instance
    :output: where to send the plot. Either a Path to save to or an existing axes
    :wrap_phase: where the model was wrapped
    """
    # If the output is a path then we need to set up a plot and axes which we'll
    # save to the output file, otherwise we'll plot to it as axes
    if isinstance(output, Path):
        fig = plt.figure(figsize=(6, 4), constrained_layout=True, dpi=100)
        ax = fig.add_subplot(111)
    else:
        ax = output
        output = None

    # Will error if nothing found
    instance = next(inspect_dataset(dataset_files, [identifier]), None)
    if not instance:
        raise KeyError(f"No match on identifier '{identifier}'")

    # Only the mags are stored in the dataset. Infer the x/phase data
    (_, _, model_feature, _) = instance
    phases = np.linspace(wrap_phase-1, wrap_phase, len(model_feature))
    ax.scatter(x=phases, y=model_feature, marker=".", s=0.25)

    # We'll rely on the caller to config the output if it's an Axes
    if output:
        ax.invert_yaxis()
        ax.set_xlabel("Phase")
        ax.set_ylabel("Relative magnitude (mag)")

        ymin, ymax = ax.get_ylim()
        ax.vlines([0.0, 0.5], ymin, ymax, linestyles="--", color="k", lw=.5, alpha=.25)

        x_ticks = [0.0, 0.5]
        ax.set_xticks(x_ticks, minor=False)
        ax.set_xticklabels(x_ticks)
        start = round(min(phases), 1)
        ax.set_xticks(np.arange(start, start + 1.0 if start < 0.0 else 1.1, 0.1), minor=True)
        ax.tick_params(axis="both", which="both", direction="in",
                       top=True, bottom=True, left=True, right=True)

        fig.savefig(output)
        plt.close()


def sector_config_from_target(sector: int, target_cfg: Dict[str, any]) -> Dict[str, any]:
    """
    Get the sector specific config from the passed target config. The sector config
    is the target config overlaid with the sector config so the sector provides overrides.
    """
    sector_cfg = {
        **target_cfg.copy(),
        **target_cfg["sectors"][f"{sector}"]
    }
    sector_cfg.pop("sectors", None)
    return sector_cfg


def prepare_lc_for_target_sector(target: str,
                                 sector: int,
                                 config: Dict[str, any],
                                 fits_dir: Path,
                                 verbose: bool=True) -> LightCurve:
    """
    Will find and load the requested target/sector lightcurve then mask, bin and
    append the rectified delta_mag and delta_mag_err columns. It will read the
    information required for these steps from the passed sector config.
    """
    lc = lightcurve.find_lightcurves(target,
                                     fits_dir,
                                     [sector],
                                     config.get("mission", "TESS"),
                                     config.get("author", "SPOC"),
                                     config.get("exptime", "short"),
                                     config.get("flux_column", "sap_flux"),
                                     config.get("quality_bitmask", "default"),
                                     verbose=verbose)[0]

    mask_time_ranges = config.get("quality_masks", None) or []
    lc = lightcurve.apply_quality_masks(lc, mask_time_ranges, verbose)

    time_bin_seconds = (config.get("bin_time", None) or 0) * u.s
    if time_bin_seconds > 0 * u.s:
        lightcurve.bin_lightcurve(lc, time_bin_seconds, verbose)

    lightcurve.append_magnitude_columns(lc, "delta_mag", "delta_mag_err")
    lc["delta_mag"] -= lightcurve.fit_polynomial(lc.time,
                                                 lc["delta_mag"],
                                                 config.get("detrend_order", 2),
                                                 config.get("detrend_iterations", 2),
                                                 config.get("detrend_sigma_clip", 1.0),
                                                 verbose=verbose)
    return lc
