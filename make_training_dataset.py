""" Script for generating the full formal training dataset. """
import os
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np

# pylint: disable=no-member
import astropy.units as u

# Tell the libraries where the JKTEBOP executable lives.
# The conda yaml based env sets this but it's not set for venvs.
if not "JKTEBOP_DIR" in os.environ:
    os.environ["JKTEBOP_DIR"] = "~/jktebop/"

# pylint: disable=wrong-import-position
# Put these after the above environ statements so the values are picked up if needed
from traininglib import datasets, plots

from ebop_maven.libs import orbital
from ebop_maven.libs.tee import Tee

# Creating a dataset is a two step process;
#   1. create a set of csv files
#   2. from these build a corresponding set of tensorflow (tfrecord) dataset files
#
# The reason for the split are;
#   - it allows for the mulitple datasets to be created from a consistent set of
#     instances (the csvs). This is useful during training/hyperparameter tuning
#     by allow what's in a dataset to be varied (e.g. phase shift of the lightcurves)
#   - tfrecord files are not easily readable so the csv files are useful for
#     access to the original parameters of an instance. You could consider the
#     dataset as the compiled output with the csvs being the source
#   - it's a convenient break in the process

DATASET_SIZE = 250000
RESUME = False
dataset_dir = Path(f"./datasets/formal-training-dataset-{DATASET_SIZE // 1000}k/")
dataset_dir.mkdir(parents=True, exist_ok=True)


def generate_instances_from_distributions(instance_count: int, label: str, verbose: bool=False):
    """
    Generates the requested number of instances by picking from random distributions
    over the JKTEBOP parameter range.

    :instance_count: the number of systems to generate
    :verbose: whether to print out verbose progress/diagnostics information
    :returns: a generator over the dictionaries, one per system
    """
    # pylint: disable=too-many-locals, invalid-name
    generated_counter = 0
    usable_counter = 0
    set_id = label.replace("trainset", "")

    while usable_counter < instance_count:
        while True: # imitate "loop and a half" / "repeat ... until" logic
            # These are the "label" params for which we have defined distributions
            rA_plus_rB  = np.random.uniform(low=0.001, high=0.45001)
            k           = np.random.normal(loc=0.8, scale=0.4)
            inc         = np.random.uniform(low=50., high=90.00001) * u.deg
            J           = np.random.normal(loc=0.8, scale=0.4)

            # We need a version of JKTEBOP which supports negative L3 input values
            # (not so for version 43) in order to train a model to predict L3.
            L3          = np.random.normal(0., 0.1)
            L3          = 0 # continue to override until revised JKTEBOP released

            # The qphot mass ratio (MB/MA) affects the lightcurve with the ellipsoidal effect from
            # distortion of the stars' shape. Set to -100 to force spherical stars or derive a value
            # Here we are using the k-q relations of Demircan & Kahraman (1991, Ap&SS, 181, pp320);
            #   - both stars <1.66 M_sun: k=q^0.935, so q=k^1.07
            #   - both stars >1.66 M_sun: k=q^0.542, so q=k^1.85
            #   - low-mass MB / high-mass MA: k=q^0.724, so q=k^1.38
            #   - general/single empirical fit; R=1.01*M^0.724 (0.1 < M < 18.1 M_sun), so q~k^1.4
            # Various combinations tried and the most effective Model found when trained on the
            # last of these relations "spiced" with a normal distribution.
            qphot       = k**np.random.normal(1.4, scale=0.2)

            # We generate ecc and omega (argument of periastron) from appropriate distributions.
            # They're not used directly as labels, but they make up ecosw and esinw which are.
            ecc         = np.abs(np.random.normal(loc=0.0, scale=0.2))
            omega       = np.random.uniform(low=0., high=360.) * u.deg

            # Now we can calculate the derived values, sufficient to check we've a usable system
            inc_rad     = inc.to(u.rad).value
            omega_rad   = omega.to(u.rad).value
            esinw       = ecc * np.sin(omega_rad)
            ecosw       = ecc * np.cos(omega_rad)
            rA          = rA_plus_rB / (1 + k)
            rB          = rA_plus_rB - rA
            imp_prm     = orbital.impact_parameter(rA, inc, ecc, None, esinw,
                                                   orbital.EclipseType.BOTH)

            generated_counter += 1
            inst_id = f"{set_id}/{generated_counter:06d}"
            if datasets.is_usable_system(rA, rB, J, qphot, ecc, inc, imp_prm, r_limit=0.23):
                break

        # Create the pset dictionary.
        yield {
            "id":           inst_id,

            # Basic system params for generating the model light-curve
            # The keys (& those for LD below) are those expected by make_dateset
            "rA_plus_rB":   rA_plus_rB,
            "k":            k,
            "inc":          inc.to(u.deg).value, 
            "qphot":        qphot,
            "ecosw":        ecosw,
            "esinw":        esinw,
            "J":            J,
            "L3":           L3,

            **datasets.default_limb_darkening_params,

            # Further params for potential use as labels/features
            "sini":         np.sin(inc_rad),
            "cosi":         np.cos(inc_rad),
            "rA":           rA,
            "rB":           rB,
            "ecc":          ecc,
            "omega":        omega.to(u.deg).value,
            "bP":           imp_prm[0],
            "bS":           imp_prm[1],                      
        }

        usable_counter += 1
        if verbose and usable_counter % 100 == 0:
            print(f"{label}: Generated {usable_counter:,} usable",
                    f"instances from {generated_counter:,} distinct configurations.")


# ------------------------------------------------------------------------------
# Makes the formal training dataset based on the above generator function which
# samples parameter distributions over JKTEBOP's usable range.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    with redirect_stdout(Tee(open(dataset_dir / "trainset.log",
                                  "w",
                                  encoding="utf8"))):
        datasets.generate_dataset_csvs(instance_count=DATASET_SIZE,
                                       file_count=DATASET_SIZE // 10000,
                                       output_dir=dataset_dir,
                                       generator_func=generate_instances_from_distributions,
                                       file_pattern="trainset{0:03d}.csv",
                                       verbose=True,
                                       simulate=False)

    plots.plot_trainset_histograms(dataset_dir, dataset_dir / "train-histogram-full.png", cols=3)
    plots.plot_trainset_histograms(dataset_dir, dataset_dir / "train-histogram-main.eps", cols=2,
                                   params=["rA_plus_rB", "k", "J", "inc", "ecosw", "esinw"])

    with redirect_stdout(Tee(open(dataset_dir / "dataset.log",
                                  "a" if RESUME else "w",
                                  encoding="utf8"))):
        datasets.make_dataset_files(trainset_files=sorted(dataset_dir.glob("trainset*.csv")),
                                    output_dir=dataset_dir,
                                    valid_ratio=0.2,
                                    test_ratio=0,
                                    resume=RESUME,
                                    max_workers=5,
                                    verbose=True,
                                    simulate=False)
