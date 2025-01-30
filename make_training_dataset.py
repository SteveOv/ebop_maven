""" Script for generating the full formal training dataset. """
import os
import sys
from pathlib import Path
from contextlib import redirect_stdout
import hashlib

import numpy as np

from deblib import orbital, limb_darkening

# Tell the libraries where the JKTEBOP executable lives.
# The conda yaml based env sets this but it's not set for venvs.
# pylint: disable=wrong-import-position
if not "JKTEBOP_DIR" in os.environ:
    os.environ["JKTEBOP_DIR"] = "~/jktebop/"

from traininglib import datasets, plots
from traininglib.tee import Tee

# Optional filtering/processing on generated LCs before acceptance
MIN_ECLIPSE_DEPTH = None
SWAP_IF_DEEPER_SECONDARY = False

# By splitting the dataset over multiple files we have the option of using a subset of the dataset
# using a wildcard match, for example "trainset00?.tfrecord" picks up the first 10 files only.
DATASET_SIZE = 500000
FILE_COUNT = DATASET_SIZE // 10000
FILE_PREFIX = "trainset"
dataset_dir = Path(f"./datasets/formal-training-dataset-{DATASET_SIZE // 1000}k/")
dataset_dir.mkdir(parents=True, exist_ok=True)

# max fractional radius: JKTEBOP unsuited to close binaries. As a "rule of thumb" the cut-off is
# at ~0.2. We go further so as to train a model which will be able to predict upto and beyond this.
MAX_FRACTIONAL_R = 0.23

# Useable "general use" limb-darkening algo and coefficients.
# Centred within the distribution of coefficients on bulk of M-S stars and logg 3.0 to 5.0.
# We plot coefficients.png showing chosen values over the trails of coeffs (by Teff) for each logg.
fixed_ld_params = [{
    "LDA": "quad",  "LDB": "quad",
    "LDA1": 0.38,   "LDB1": 0.38,
    "LDA2": 0.22,   "LDB2": 0.22
}]

def generate_instances_from_distributions(label: str):
    """
    Generates system instances by picking from random distributions over the
    JKTEBOP parameter range.

    :label: a useful label to use within messages
    :returns: a generator over instance parameter dictionaries, one per system
    """
    # pylint: disable=too-many-locals, invalid-name
    generated_counter = 0
    set_id = ''.join(filter(str.isdigit, label))

    # Don't use the built-in hash() function; it's not consistent across processes!!!
    seed = int.from_bytes(hashlib.shake_128(label.encode("utf8")).digest(8))
    rng = np.random.default_rng(seed)

    while True: # infinite loop; we will continue to yield new instances until generator is closed
        # The "label" params are rA_plus_rB, k, J, ecosw, esinw and bP (or inc, depending on model)

        # Extending range of rA_plus_rB beyond 0.2+0.2 we need to be able to predict systems where
        # JKTEBOP may not be suitable (we may not know this unless the predictions are reliable).
        # The distributions for k and J are chosen through testing, being those which yield a
        # model capable of making good predictions over the various testing datasets.
        rA_plus_rB  = rng.uniform(low=0.001, high=0.45001)
        k           = rng.normal(loc=0.5, scale=0.8)
        rA          = rA_plus_rB / (1 + k)              # Not used directly as labels, but useful
        rB          = rA_plus_rB - rA

        # Simple uniform dist for inc (JKTEBOP bottoms out at 50 deg)
        inc         = rng.uniform(low=50., high=90.00001)   # deg
        J           = rng.normal(loc=0.5, scale=0.8)

        # We need a version of JKTEBOP which supports negative L3 input values
        # (not so for version 43) in order to train a model to predict L3.
        L3          = rng.normal(0., 0.1)
        L3          = 0 # continue to override until revised JKTEBOP released

        # The qphot mass ratio value (MB/MA) affects the lightcurve via the ellipsoidal effect
        # due to distortion of the stars' shape. Set to -100 to force spherical stars or derive
        # a value from other params. We're using the k-q relations of Demircan & Kahraman (1991)
        # Both <1.66 M_sun (k=q^0.935), both >1.66 M_sun (k=q^0.542), MB-low/MA-high (k=q^0.724)
        # and approx' single rule is k = q^0.715 which we use here (tests find this works best).
        qphot       = rng.normal(loc=k**1.4, scale=0.3) if k > 0 else 0

        # We generate ecc and omega (argument of periastron) from appropriate distributions.
        # They're not used directly as labels, but they make up ecosw and esinw which are.
        # Eccentricity is uniform selection, restricted with eqn 5 of Wells & Prsa (2024) which
        # reduces the max eccentricity with the separation plus 10% are fixed at 0 to ensure
        # sufficient examples. This trains better than simple uniform or normal distributions tried.
        ecc         = rng.choice([0, rng.uniform(low=0., high=1-(1.5*rA_plus_rB))], p=[0.1, 0.9])
        omega       = rng.uniform(low=0., high=360.)        # deg

        # Now we can calculate the derived values, sufficient to check we've a usable system
        inc_rad     = np.radians(inc)
        omega_rad   = np.radians(omega)
        esinw       = ecc * np.sin(omega_rad)
        ecosw       = ecc * np.cos(omega_rad)
        bp          = orbital.impact_parameter(rA, inc, ecc, esinw, False)
        bs          = orbital.impact_parameter(rA, inc, ecc, esinw, True)

        # Create the pset dictionary.
        generated_counter += 1
        inst_id = f"{set_id}/{generated_counter:06d}"
        yield {
            "id":           inst_id,

            # Basic system params for generating the model light-curve
            # The keys (& those for LD below) are those expected by make_dateset
            "rA_plus_rB":   rA_plus_rB,
            "k":            k,
            "inc":          inc, 
            "qphot":        qphot,
            "ecosw":        ecosw,
            "esinw":        esinw,
            "J":            J,
            "L3":           L3,

            **fixed_ld_params[generated_counter % len(fixed_ld_params)],

            # Further params for potential use as labels/features
            "sini":         np.sin(inc_rad),
            "cosi":         np.cos(inc_rad),
            "rA":           rA,
            "rB":           rB,
            "ecc":          ecc,
            "omega":        omega,
            "bP":           bp,
            "bS":           bs,
            "phiS":         orbital.phase_of_secondary_eclipse(ecosw, ecc),
            "dS_over_dP":   orbital.ratio_of_eclipse_duration(esinw),
        }


def is_usable_instance(k: float=0, J: float=0, qphot: float=0, ecc: float=0,
                       bP: float= None, bS: float=None,
                       rA: float=1., rB: float=1., inc: float=0,
                       **_ # Used to ignore any unexpected **params
                       ) -> bool:
    """
    Checks various parameter values to decide whether this represents a usable instance.
    Checks on;
    - is system physically plausible
    - will it generate eclipses
    - is it suitable for modelling with JKTEBOP
    """
    # pylint: disable=invalid-name, too-many-arguments, unused-argument
    usable = False

    # Use invalid values as defaults so that if any are missing we fail
    # Physically plausible (qphot of -100 is a magic number to force spherical)
    usable = k > 0 and J > 0 and (qphot > 0 or qphot == -100) and ecc < 1

    # Will eclipse
    if usable:
        usable = all(b is not None and b <= 1 + k for b in [bP, bS])

    # Compatible with JKTEBOP restrictions
    # Hard restrictions of rA+rB < 0.8 (covered by MAX_FRACTIONAL_R), inc > 50, k <= 100
    if usable:
        usable = rA <= MAX_FRACTIONAL_R and rB <= MAX_FRACTIONAL_R and inc > 50 and k <= 100
    return usable


# ------------------------------------------------------------------------------
# Makes the formal training dataset based on the above generator function which
# samples parameter distributions over JKTEBOP's usable range.
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    if (dataset_dir/"dataset.log").exists():
        response =  input("\nFiles exist for this dataset. Continue and overwrite y/N? ")
        if response.strip().lower() not in ["y", "yes"]:
            sys.exit()

    with redirect_stdout(Tee(open(dataset_dir/"dataset.log", "w", encoding="utf8"))):
        # Plot the general purpose quad/TESS coefficients against values for Z=0 & various Teff/logg
        table = limb_darkening._quad_ld_coeffs_table("TESS") # pylint: disable=protected-access
        fig = plots.plot_limb_darkening_coeffs(table[table["Z"]==0], fixed_ld_params,
                title=r"TESS LD coefficients v $T_{\rm eff}$ at $Z=0$ (Claret, 2018)",
                xlabel="linear coefficient", ylabel="quadratic coefficient", legend_loc="best")
        fig.savefig(dataset_dir / "limb_darkening.png", dpi=150)
        fig.clf()

        datasets.make_dataset(instance_count=DATASET_SIZE,
                              file_count=FILE_COUNT,
                              output_dir=dataset_dir,
                              generator_func=generate_instances_from_distributions,
                              check_func=is_usable_instance,
                              min_eclipse_depth=MIN_ECLIPSE_DEPTH,
                              swap_if_deeper_secondary=SWAP_IF_DEEPER_SECONDARY,
                              file_prefix=FILE_PREFIX,
                              valid_ratio=0.2,
                              test_ratio=0,
                              max_workers=5,
                              save_param_csvs=True,
                              verbose=True,
                              simulate=False)

        # Histograms are generated from the CSV files as they cover params not saved to tfrecord
        csvs = sorted(dataset_dir.glob(f"**/{FILE_PREFIX}*.csv"))
        plots.plot_dataset_histograms(csvs, cols=5).savefig(dataset_dir/"train-histogram-full.png")
        plots.plot_dataset_histograms(csvs, ["rA_plus_rB", "k", "J", "inc", "ecosw", "esinw"],
                                      cols=2).savefig(dataset_dir/"train-histogram-main.pdf")

        # Simple diagnostic plot of the mags feature of a small sample of the instances.
        print("Plotting a sample of the set's mags features")
        dataset_files = sorted(dataset_dir.glob(f"**/training/{FILE_PREFIX}000.tfrecord"))
        fig = plots.plot_dataset_instance_mags_features(dataset_files, cols=5, max_instances=50)
        fig.savefig(dataset_dir / "sample.png", dpi=150)
        fig.clf()
