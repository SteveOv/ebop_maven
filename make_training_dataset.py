""" Script for generating the full formal training dataset. """
import os
import sys
from pathlib import Path
from contextlib import redirect_stdout
import hashlib
from inspect import getsource

import numpy as np
import matplotlib.pyplot as plt

from deblib import orbital, limb_darkening

# Tell the libraries where the JKTEBOP executable lives.
# The conda yaml based env sets this but it's not set for venvs.
# pylint: disable=wrong-import-position
if not "JKTEBOP_DIR" in os.environ:
    os.environ["JKTEBOP_DIR"] = "~/jktebop/"

from traininglib import datasets, plots
from traininglib.tee import Tee

# By splitting the dataset over multiple files we have the option of using a subset of the dataset
# using a wildcard match, for example "trainset00?.tfrecord" picks up the first 10 files only.
DATASET_SIZE = 500000
FILE_COUNT = DATASET_SIZE // 10000
FILE_PREFIX = "trainset"
dataset_dir = Path(f"./datasets/formal-training-dataset-{DATASET_SIZE // 1000}k/")
dataset_dir.mkdir(parents=True, exist_ok=True)

# Optional filtering/processing on generated LCs before acceptance
SWAP_IF_DEEPER_SECONDARY = True

# Control whether static augmentations (noise, roll & y-shift) are applied to mags features when the
# dataset is created. They will never be applied to train instances if IGNORE_AUGS_ON_TRAIN is True.
APPLY_STATIC_AUGS = True
IGNORE_AUGS_ON_TRAIN = True

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
    rng_aug = np.random.default_rng(seed) # Separate so augs on/off doesn't affect other params

    while True: # infinite loop; we will continue to yield new instances until generator is closed
        # The "label" params are rA_plus_rB, k, J, ecosw, esinw and bP (or inc, depending on model)

        # Extending range of rA_plus_rB beyond 0.2+0.2 we need to be able to predict systems where
        # JKTEBOP may not be suitable (we may not know this unless the predictions are reliable).
        rA          = rng.uniform(low=0.001, high=MAX_FRACTIONAL_R)
        rB          = rng.uniform(low=0.001, high=MAX_FRACTIONAL_R)
        k           = rB / rA
        rA_plus_rB  = rA + rB

        J           = rng.uniform(low=0.001, high=1.0) / rng.uniform(low=0.001, high=1.0)

        # Simple uniform dist for cos(inc) (JKTEBOP bottoms out at 50 deg)
        cosi        = rng.uniform(high=0.643, low=0.) # uniform in cosi between ~50 & 90 deg
        inc_rad     = np.arccos(cosi)
        inc         = np.degrees(inc_rad)   # deg

        # We need a version of JKTEBOP which supports negative L3 input values
        # (not so for version <= 43) in order to train a model to predict L3.
        L3          = 0 # continue to override until revised JKTEBOP released

        # The qphot mass ratio value (MB/MA) affects the lightcurve via the ellipsoidal effect
        # due to distortion of the stars' shape. Set to -100 to force spherical stars or derive
        # a value from other params. We're using the k-q relations of Demircan & Kahraman (1991);
        # their approximate single rule is k=q^0.715 which we use here (tests find this works best).
        qphot       = k**1.4

        # We generate ecc and omega (argument of periastron) from simple distributions.
        # They're not used directly as labels, but they make up ecosw & esinw which are.
        ecc         = abs(rng.normal(loc=0, scale=0.5))
        omega       = rng.uniform(low=0., high=360.)        # deg

        generated_counter += 1
        if 0 <= ecc < 1: # Skip inst if ecc invalid as it may break some downstream calculations
            omega_rad   = np.radians(omega)
            esinw       = ecc * np.sin(omega_rad)
            ecosw       = ecc * np.cos(omega_rad)

            yield {
                "id":           f"{set_id}/{generated_counter:06d}",

                # Basic system params for generating the model light-curve
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
                "cosi":         cosi,
                "rA":           rA,
                "rB":           rB,
                "ecc":          ecc,
                "omega":        omega,
                "bP":           orbital.impact_parameter(rA, inc, ecc, esinw, False),
                "bS":           orbital.impact_parameter(rA, inc, ecc, esinw, True),
                "phiS":         orbital.phase_of_secondary_eclipse(ecosw, ecc),
                "dS_over_dP":   orbital.ratio_of_eclipse_duration(esinw),

                # Optional static augmentations applied to the mags_features generated. Even if set,
                # they won't be applied to training instances if IGNORE_AUGS_ON_TRAIN is True.
                "noise_sigma":  rng_aug.uniform(low=1e-4, high=3e-2) if APPLY_STATIC_AUGS else None,
                "phase_shift":  rng_aug.normal(loc=0, scale=.03) if APPLY_STATIC_AUGS else None,
                "mag_shift":    rng_aug.normal(loc=0, scale=.03) if APPLY_STATIC_AUGS else None,
            }


def is_usable_instance(rA_plus_rB: float, k: float, J: float, qphot: float, ecc: float,
                       L3: float, bP: float, bS: float, rA: float, rB: float, inc: float,
                       phiP: float=0, phiS: float=0.5, depthP: float=100, depthS: float=100,
                       **_ # Used to ignore any unexpected **params
                       ) -> bool:
    """
    Checks various parameter values to decide whether this represents a usable instance.
    Checks on;
    - is system physically plausible
    - will it generate eclipses and are they sufficiently prominent to be usable
    - is it suitable for modelling with JKTEBOP
    """
    # pylint: disable=invalid-name, too-many-arguments, unused-argument
    # Physically plausible
    usable = 0 <= ecc < 1 and -1 < L3 < 1

    # Will eclipse, and that they are sufficiently prominent to be useful for training
    if usable:
        usable = all(b is not None and b <= 1 + k for b in [bP, bS]) \
            and min(depthP, depthS) >= 0.010

    # Compatible with JKTEBOP restrictions (qphot of -100 is a magic number to force spherical)
    # Hard restrictions of rA+rB < 0.8 (covered by MAX_FRACTIONAL_R) etc...
    if usable:
        usable = rA <= MAX_FRACTIONAL_R and rB <= MAX_FRACTIONAL_R \
            and 50 <= inc <= 140 and 0.01 <= k <= 100 and 0.001 <= J <= 1000 \
            and (qphot == -100 or 0.001 <= qphot <= 1000)
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
        plt.close(fig)

        code_file = dataset_dir / "parameter-distributions.txt"
        with code_file.open("w", encoding="utf8") as of:
            of.write(f"SWAP_IF_DEEPER_SECONDARY = {SWAP_IF_DEEPER_SECONDARY}\n")
            of.write(f"APPLY_STATIC_AUGS = {APPLY_STATIC_AUGS}\n")
            of.write(f"IGNORE_AUGS_ON_TRAIN = {IGNORE_AUGS_ON_TRAIN}\n")
            of.write(f"MAX_FRACTIONAL_R = {MAX_FRACTIONAL_R}\n\n")
            of.write(getsource(generate_instances_from_distributions))
            of.write("\n\n")
            of.write(getsource(is_usable_instance))
        print(f"Saved copies of the param distribution & constraint functions to {code_file.name}")

        datasets.make_dataset(instance_count=DATASET_SIZE,
                              file_count=FILE_COUNT,
                              output_dir=dataset_dir,
                              generator_func=generate_instances_from_distributions,
                              check_func=is_usable_instance,
                              swap_if_deeper_secondary=SWAP_IF_DEEPER_SECONDARY,
                              ignore_augs_on_train=IGNORE_AUGS_ON_TRAIN,
                              file_prefix=FILE_PREFIX,
                              valid_ratio=0.2,
                              test_ratio=0,
                              max_workers=5,
                              save_param_csvs=True,
                              verbose=True,
                              simulate=False)

        # Histograms are generated from the CSV files as they cover params not saved to tfrecord
        csv = sorted(dataset_dir.glob(f"**/{FILE_PREFIX}*.csv"))
        fig = plots.plot_dataset_histograms(csv, cols=5)
        fig.savefig(dataset_dir / "train-histogram-full.png", dpi=150)
        plt.close(fig)
        fig = plots.plot_dataset_histograms(csv, ["rA_plus_rB", "k", "J", "inc", "ecosw", "esinw"],
                                            cols=2, ignore_outliers=True)
        fig.savefig(dataset_dir / "train-histogram-main.pdf")
        plt.close(fig)

        # Simple diagnostic plot of the mags feature of a small sample of the instances.
        for dataset_file in sorted(dataset_dir.glob(f"**/{FILE_PREFIX}000.tfrecord")):
            print(f"Plotting a sample of the {dataset_file.parent.name} subset's mags features")
            fig = plots.plot_dataset_instance_mags_features([dataset_file], mags_wrap_phase=1,
                                                            cols=5, max_instances=50)
            fig.savefig(dataset_dir / f"sample-{dataset_file.parent.name}.png", dpi=150)
            fig.clf()
