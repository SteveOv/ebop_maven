"""
Functions for generating binary system instances and writing their parameters to trainset csv files.
"""
# Use ucase variable names where they match the equivalent symbol and pylint can't find units alias
# pylint: disable=invalid-name, no-member, too-many-arguments, too-many-locals

from typing import Callable, Generator
import sys
from pathlib import Path
from timeit import default_timer
from datetime import timedelta, datetime
import traceback
from multiprocessing import Pool
import csv
import hashlib

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import tensorflow as tf

# Hack so that this module can see the ebop_maven package and below
# pylint: disable=wrong-import-order, wrong-import-position
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
        (inst_count, file_ix, output_dir, generator_func, check_func, valid_ratio, test_ratio,
         file_prefix, swap_if_deeper_secondary, save_param_csvs, verbose, simulate)
            for (file_ix, inst_count) in enumerate(file_inst_counts)
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
                    ds_writers[subset_ix] = tf.io.TFRecordWriter(f"{ds_subset_file}",
                                                                 deb_example.ds_options)
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

                is_usable = check_func(**params)
                if is_usable:
                    model_data = jktebop.generate_model_light_curve(file_prefix, **params)

                    if swap_if_deeper_secondary:
                        # Check the primary eclipse is the deeper. If not, we can swap the
                        # components and roll the mags to move the secondary eclipse to phase zero.
                        max_ix = np.argmax(model_data["delta_mag"])
                        if 99 < max_ix < len(model_data) -99:
                            _swap_instance_components(params)
                            is_usable = check_func(**params)
                            if is_usable:
                                model_data["delta_mag"] = np.roll(model_data["delta_mag"], -max_ix)
                                swap_count += 1
                                params["swapped"] = 1

                if is_usable:
                    # Occasionally, JKTEBOP outputs NaN for a mag. Handle this by setting it to zero
                    if np.isnan(np.min(model_data["delta_mag"])):
                        if verbose:
                            print(f"{file_stem}[{inst_id}]: Replacing NaN/Inf in processed LC.")
                        np.nan_to_num(x=model_data["delta_mag"], copy=False)

                    # Optionally, add Gaussian flux noise based on the instance's apparent magnitude
                    apparent_mag = params.get("apparent_mag", None)
                    if apparent_mag:
                        # The SNR is based on linear regression fit of Álvarez et al. (2024) Table 2
                        # then noise sigma from re-arranging their eqn 7; SNR = 10*log_10(mu/sigma)
                        # As the fluxes we derive will be normalized assume the mean ~ 1.
                        snr = np.add(np.multiply(-2.32, apparent_mag), 59.4)
                        noise_sigma = np.divide(1, np.power(10, np.divide(snr, 10)))
                        if noise_sigma:
                            # We apply the noise to fluxes, so revert delta mags to normalized flux
                            fluxes = np.power(10, np.divide(model_data["delta_mag"], -2.5))
                            noise = rng.normal(0., scale=noise_sigma, size=len(fluxes))
                            model_data["delta_mag"] = np.multiply(-2.5, np.log10(fluxes + noise))

                    phase_shift = params.get("phase_shift", None)
                    if phase_shift:
                        # Add a roll to the model data equal the phase shift
                        shift = int(len(model_data) * phase_shift)
                        model_data["delta_mag"] = np.roll(model_data["delta_mag"], shift)

                    mag_shift = params.get("mag_shift", None)
                    if mag_shift:
                        # Adds an optional shift the the magnitudes up/down; shifts the zero point
                        model_data["delta_mag"] += mag_shift

                    # We store mags_features for various supported bins values
                    mags_features = {}
                    interp = interp1d(model_data["phase"], model_data["delta_mag"], interp_kind)
                    for mag_name, mags_bins in deb_example.stored_mags_features.items():
                        # Ensure we don't waste a bin repeating the value for phase 0.0 & 1.0
                        new_phases = np.linspace(0., 1., mags_bins + 1)[:-1]
                        bin_model_data = np.array([new_phases, interp(new_phases)], dtype=np.double)
                        mags_features[mag_name] = bin_model_data[1]

                    # Any extra features which may be used for predictions alongside the LC.
                    extra_features = {
                        "phiS": params.get("phiS", None) \
                                or orbital.secondary_eclipse_phase(params["ecosw"], params["ecc"]),
                        "dS_over_dP": params.get("dS_over_dP", None) \
                                or orbital.ratio_of_eclipse_duration(params["esinw"]),
                    }

                    # Write the appropriate dataset train/val/test file, based on inst/file indices
                    row = deb_example.serialize(inst_id, params, mags_features, extra_features)
                    ds_writers[inst_file_ixs[inst_ix]].write(row)

                    if csv_file:
                        write_param_sets_to_csv(csv_file, [params], append=True)

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

def write_param_sets_to_csv(file_name: Path,
                            param_sets: list[dict],
                            field_names: list[any] = None,
                            append: bool=False) -> None:
    """
    Writes the list of parameter set dictionaries to a csv file.

    :file_name: the full name of the file to create or overwrite.
    :param_sets: the list of dictionaries to write out.
    :field_names: the list of fields to write, in the required order. If
    None, the field_names will be read from the first item in param_sets
    :mode: the mode to open the file, "w" write or "a" append
    """
    # This data is saved in an intermediate form as we've yet to
    # generate the actual light-curves. We use csv, as this is
    # easily read/consumed by apps for reviewing and the
    # tensorflow data API for subsequent processing.
    if field_names is None:
        field_names = param_sets[0].keys()
    with open(file_name, mode="a+" if append else "w", encoding="UTF8", newline='') as f:
        dw = csv.DictWriter(f,
                            field_names,
                            quotechar="'",
                            quoting=csv.QUOTE_NONNUMERIC)
        if not append or not f.tell():
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


def get_field_names_from_csvs(file_names: list[Path]) -> list[str]:
    """
    Returns a list of the field names common to all of the passed CSV files.

    :file_names: the full names of the csv files containing the parameter sets
    :returns: a list[str] of the common names
    """
    names: list[str] = None
    for file_name in file_names:
        # pylint: disable=not-an-iterable
        with open(file_name, mode="r", encoding="utf8") as pf:
            csv_reader = csv.reader(pf, quotechar="'", quoting=csv.QUOTE_NONNUMERIC)
            this_names = next(csv_reader)

        # Modify names so that it hold the names common to both
        names = [n for n in names if n in this_names] if names is not None else this_names
    return names


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

    # These are related to the Poincare elements.
    if "phiS" in params:
        params["phiS"] = orbital.secondary_eclipse_phase(params["ecosw"], None, params["esinw"])
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
    params["bP"], params["bS"] = orbital.impact_parameter(params["rA"], params["inc"] * u.deg,
                                                          e=params.get("ecc", 0),
                                                          esinw=params["esinw"],
                                                          eclipse=orbital.EclipseType.BOTH)
