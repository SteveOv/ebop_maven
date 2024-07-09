"""
Functions for generating binary system instances and writing their parameters to trainset csv files.
"""
# Use ucase variable names where they match the equivalent symbol and pylint can't find units alias
# pylint: disable=invalid-name, no-member, too-many-arguments, too-many-locals

from typing import Callable, Generator, Iterator
from pathlib import Path
from timeit import default_timer
from datetime import timedelta
import random
import traceback
from multiprocessing import Pool
import csv

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import tensorflow as tf

# Hack so that this module can see the ebop_maven package and below
# pylint: disable=wrong-import-order, wrong-import-position
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, Path("../ebop_maven/"))
from ebop_maven import deb_example
from ebop_maven.libs import jktebop, orbital

# Useable "general use" limb-darkening algo and coefficients
# for a F-type star T_eff~7200 K and logg~4.0
default_limb_darkening_params = {
    "LDA": "quad", "LDB": "quad",
    "LDA1": 0.28,  "LDB1": 0.28,
    "LDA2": 0.22,  "LDB2": 0.22
}

# The full set of parameters available for histograms, their #bins and plot labels
histogram_params = {
    "rA_plus_rB":   (100, r"$r_{A}+r_{B}$"),
    "k":            (100, r"$k$"),
    "inc":          (100, r"$i~(^{\circ})$"),
    "sini":         (100, r"$\sin{i}$"),
    "cosi":         (100, r"$\cos{i}$"),
    "qphot":        (100, r"$q_{phot}$"),
    #"L3":           (100, r"$L_3$"), # currently always zero
    "ecc":          (100, r"$e$"),
    "omega":        (100, r"$\omega~(^{\circ})$"),
    "J":            (100, r"$J$"),
    "ecosw":        (100, r"$e\,\cos{\omega}$"),
    "esinw":        (100, r"$e\,\sin{\omega}$"),
    "rA":           (100, r"$r_A$"),
    "rB":           (100, r"$r_B$"),
    "bP":           (100, r"$b_{prim}$")
}

def generate_dataset_csvs(instance_count: int,
                   file_count: int,
                   output_dir: Path,
                   generator_func: Callable[[int, str, bool], Generator[dict[str, any], any, any]],
                   seed: float=42,
                   file_pattern: str="dataset{0:03d}.csv",
                   verbose: bool=False,
                   simulate: bool=False) -> None:
    """
    Writes trainset csv file(s) with instances created by the chosen generator_func.

    :instance_count: the number of training instances to create
    :file_count: the number of files to spread them over
    :output_dir: the directory to write the files to
    :generator_func: the function to call to generate the required number of systems which must have
    arguments (instance_count: int, file_stem: str, verbose: bool) and return a Generator[dict[str]]
    :seed: random seed to ensure repeatability
    :file_pattern: naming pattern for the csv files. Must have a single integer format placeholder.
    :verbose: whether to print verbose progress/diagnostic messages
    :simulate: whether to simulate the process, skipping only file/directory actions
    """
    if not output_dir:
        output_dir = Path("~/datasets/formal-trainset/").expanduser()
    if not simulate:
        output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"""
Generate & write dEB system parameters CSVs for required number of instances
--------------------------------------------------------=-------------------
The number of instances to generate:    {instance_count:,} across {file_count} file(s)
The instance generator function is:     {generator_func.__name__}
The parameter CSVs will be written to:  {output_dir}
The parameter CSVs will be named:       {file_pattern}
The random seed to use for selections:  {seed}\n""")
        if simulate:
            print("Simulate requested so no files will be written.\n")

    if verbose:
        start_time = default_timer()

    # This is reasonably quick so there's no need to use a process pool here.
    # Also, a pool may mess with the repeatability of the psuedo random behaviour.
    np.random.seed(seed)
    for ix, file_inst_count in enumerate(_calculate_file_splits(instance_count, file_count), 1):
        out_file = output_dir / file_pattern.format(ix)
        psets = [*generator_func(file_inst_count, out_file.stem, verbose)]
        if verbose:
            print(f"{'Simulated s' if simulate else 'S'}aving {len(psets)} inst(s) to {out_file}")
        if not simulate:
            write_param_sets_to_csv(out_file, psets)

    if verbose:
        print(f"\nFinished. The time taken was {timedelta(0, round(default_timer()-start_time))}.")


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
    for counter, params in enumerate(read_param_sets_from_csv(trainset_file), 1):
        params.setdefault("gravA", 0.)
        params.setdefault("gravB", 0.)

        # Large negative values force JKTEBOP task2 to calculate the reflection coefficients
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

            # These are the extra features which may be used for predictions alongside the LC.
            extra_features = {
                "phiS": orbital.secondary_eclipse_phase(params["ecosw"], params["ecc"]),
                "dS_over_dP": orbital.ratio_of_eclipse_duration(params["esinw"]),
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
        msg = None
        short_name = f"{subset_file.parent.name}/{subset_file.name}"
        if simulate:
            msg = f"{label}: Simulated saving {subset_count} {subset} instance(s) to {short_name}"
        elif subset_count > 0:
            # (Over)write the subset file. Rows are selected as a contiguous block of the shuffled
            # indices. which are then re-sorted so the rows is written in the original order.
            subset_slice = slice(subset_slice_start, subset_slice_start + subset_count)
            subset_file.parent.mkdir(parents=True, exist_ok=True)
            with tf.io.TFRecordWriter(f"{subset_file}", deb_example.ds_options) as ds:
                for sorted_ix in sorted(row_indices[subset_slice]):
                    ds.write(rows[sorted_ix])
            subset_slice_start = subset_slice.stop
            msg = f"{label}: Saved {subset_count} {subset} instance(s) to {short_name}"
        else:
            # Delete the existing file which may be left from previous run or we
            # will be left with too many rows, and duplicates, over the subsets.
            subset_file.unlink(missing_ok=True)
        if verbose and msg:
            print(msg)


def write_param_sets_to_csv(file_name: Path,
                            param_sets: list[dict],
                            field_names: list[any] = None) -> None:
    """
    Writes the list of parameter set dictionaries to a csv file.

    :file_name: the full name of the file to create or overwrite.
    :param_sets: the list of dictionaries to write out.
    :field_names: the list of fields to write, in the required order. If
    None, the field_names will be read from the first item in param_sets
    """
    # This data is saved in an intermediate form as we've yet to
    # generate the actual light-curves. We use csv, as this is
    # easily read/consumed by apps for reviewing and the
    # tensorflow data API for subsequent processing.
    if field_names is None:
        field_names = param_sets[0].keys()
    with open(file_name, mode="w", encoding="UTF8", newline='') as f:
        dw = csv.DictWriter(f,
                            field_names,
                            quotechar="'",
                            quoting=csv.QUOTE_NONNUMERIC)
        dw.writeheader()
        dw.writerows(param_sets)


def read_param_sets_from_csv(file_name: Path) -> Generator[dict, any, None]:
    """
    Reads a list of parameter set dictionaries from a csv file,
    as created by write_param_sets_to_csv()

    :file_name: the full name of the csv file containing the parameter sets
    :returns: a generator for the dictionaries
    """
    with open(file_name, mode="r", encoding="UTF8") as pf:
        dr = csv.DictReader(pf, quotechar="'", quoting=csv.QUOTE_NONNUMERIC)
        for param_set in dr:
            yield param_set


def read_param_sets_from_csvs(file_names: list[Path]) -> Generator[dict, any, None]:
    """
    Reads a list of parameter set dictionaries from across all of the csv files,
    as created by write_param_sets_to_csv()

    :file_names: the full names of the csv files containing the parameter sets
    :returns: a generator for the dictionaries
    """
    for file_name in file_names:
        for param_set in read_param_sets_from_csv(file_name):
            yield param_set


def is_usable_system(rA: float, rB: float, J: float, qphot: float,
                    ecc: float, inc: float, imp_params: tuple[float], eclipse_baseline=1) -> bool:
    """
    Checks various  values to decide whether this represents a usable system.
    Checks on;
    - is system physically plausible
    - will it generate eclipses
    - is it suitable for modelling with JKTEBOP
    """
    usable = False

    k = rB / rA

    # Physically plausible
    usable = k > 0 and J > 0 and qphot > 0 and ecc < 1

    # Will eclipse
    if usable:
        usable = all(b <= eclipse_baseline + k for b in imp_params)

    # Compatible with JKTEBOP restrictions
    # Soft restriction of rA & rB both < 0.2 as its model is not suited to higher
    # Hard restrictions of rA+rB<0.8 (covered by above), inc > 50
    # TODO: will need to extend this for L3 if we start to use non-Zero L3 values
    if usable:
        if isinstance(inc, u.Quantity):
            inc = inc.to(u.deg).value
        usable = rA < 0.2 and rB < 0.2 and inc > 50
    return usable


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
