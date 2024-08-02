""" Script for generating the full formal training dataset. """
import os
from pathlib import Path
from contextlib import redirect_stdout
import hashlib

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

DATASET_SIZE = 250000
FILE_PREFIX = "trainset"
dataset_dir = Path(f"./datasets/formal-training-dataset-{DATASET_SIZE // 1000}k/")
dataset_dir.mkdir(parents=True, exist_ok=True)


def generate_instances_from_distributions(label: str, verbose: bool=False):
    """
    Generates system instances by picking from random distributions over the
    JKTEBOP parameter range.

    :label: a useful label to use within messages
    :verbose: whether to print out verbose progress/diagnostics information
    :returns: a generator over instance parameter dictionaries, one per system
    """
    # pylint: disable=too-many-locals, invalid-name
    generated_counter = 0
    usable_counter = 0
    set_id = ''.join(filter(str.isdigit, label))

    # Don't use the built-in hash() function; it's not consistent across processes!!!
    seed = int.from_bytes(hashlib.shake_128(label.encode("utf8")).digest(8))
    rng = np.random.default_rng(seed)

    while True: # infinite loop; we will continue to yield new instances until closed
        while True: # imitate "loop and a half" / "repeat ... until" logic
            # These are the "label" params for which we have defined distributions
            rA_plus_rB  = rng.uniform(low=0.001, high=0.45001)
            k           = rng.normal(loc=0.8, scale=0.4)
            inc         = rng.uniform(low=50., high=90.00001) * u.deg
            J           = rng.normal(loc=0.8, scale=0.4)

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
            ecc         = np.abs(rng.normal(loc=0.0, scale=0.2))
            omega       = rng.uniform(low=0., high=360.) * u.deg

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

    with redirect_stdout(Tee(open(dataset_dir/"dataset.log", "w", encoding="utf8"))):
        datasets.make_dataset(instance_count=DATASET_SIZE,
                              file_count=DATASET_SIZE // 10000,
                              output_dir=dataset_dir,
                              generator_func=generate_instances_from_distributions,
                              file_prefix=FILE_PREFIX,
                              valid_ratio=0.2,
                              test_ratio=0,
                              max_workers=5,
                              save_param_csvs=True,
                              verbose=True,
                              simulate=False)

        # Histograms are generated from the CSV files as they cover params not saved to tfrecord
        csvs = sorted(dataset_dir.glob(f"**/{FILE_PREFIX}*.csv"))
        plots.plot_dataset_histograms(csvs, cols=3).savefig(dataset_dir/"train-histogram-full.png")
        plots.plot_dataset_histograms(csvs, ["rA_plus_rB", "k", "J", "inc", "ecosw", "esinw"],
                                      cols=2).savefig(dataset_dir/"train-histogram-main.eps")
