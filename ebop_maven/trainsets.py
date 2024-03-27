"""
Generates a synthetic trainset based on parameter distribution functions
"""
# We use some ucase variable names as they match the equivalent symbol
# pylint: disable=invalid-name

from typing import Dict, List
from pathlib import Path
from inspect import getsourcefile
from timeit import default_timer
from datetime import timedelta

import numpy as np
import astropy.units as u

from .libs import param_sets, orbital

_this_dir = Path(getsourcefile(lambda:0)).parent

# Useable "general use" limb-darkening algo and coefficients
_def_limb_darkening_params = {"LDA": "quad", "LDB": "quad",
                              "LDA1": 0.25,  "LDB1": 0.25,
                              "LDA2": 0.22,  "LDB2": 0.22 }

def write_trainset_from_distributions(instance_count: int,
                                      file_count: int,
                                      output_dir: Path,
                                      seed: float=42,
                                      verbose: bool=False,
                                      simulate: bool=False) -> None:
    """
    Writes trainset csv file(s) with instances created from random
    distributions of system parameters.

    :instance_count: the number of training instances to create
    :file_count: the number of files to spread them over
    :output_dir: the directory to write the files to
    :seed: random seed to ensure repeatability
    :verbose: whether to print verbose progress/diagnostic messages
    :simulate: whether to simulate the process, skipping only file/directory actions
    """
    if not output_dir:
        output_dir = Path("~/datasets/formal-trainset/").expanduser()
    if not simulate:
        output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"""
Generate & write dEB system parameters CSVs from random parameter distributions
-------------------------------------------------------------------------------
The number of instances to generate:    {instance_count:,} across {file_count} file(s)
The parameter CSVs will be written to:  {output_dir}
The random seed to use when selections: {seed}\n""")
        if simulate:
            print("Simulate requested so no files will be written.\n")

    if verbose:
        start_time = default_timer()

    # This is reasonably quick so there's no need to use a process pool here.
    # If we decide we need parallel running we'll have to look at setting the seed per file.
    np.random.seed(seed)
    for ix, file_inst_count in enumerate(_calculate_file_splits(instance_count, file_count), 1):
        out_file = output_dir / f"trainset{ix:02d}.csv"
        psets = [*generate_instances_from_distributions(file_inst_count, out_file.name, verbose)]
        if verbose:
            print(f"{'Simulated s' if simulate else 'S'}aving {len(psets)} system(s) to {out_file}")
        if not simulate:
            param_sets.write_param_sets_to_csv(out_file, psets)

    if verbose:
        print(f"\nFinished. The time taken was {timedelta(0, round(default_timer()-start_time))}.")


def generate_instances_from_distributions(instance_count: int, label: str, verbose: bool=False):
    """
    Generates the requested number of instances based on random distributions
    system of parameters.

    :instance_count: the number of systems to write to it
    :verbose: whether to print out verbose progress/diagnostics information
    :returns: a generator over the dictionaries, one per system
    """
    generated_counter = 0
    usable_counter = 0

    while usable_counter < instance_count:
        # These are the "label" params for which we have defined distributions
        rA_plus_rB  = np.random.uniform(low=0.001, high=0.45001)
        k           = np.random.normal(loc=0.8, scale=0.4)
        inc         = np.random.uniform(low=50., high=90.00001) * u.deg
        J           = np.random.normal(loc=0.8, scale=0.2)

        # We're once more predicting L3 as JKTEBOP is being updated to support
        # negative L3 input values (so it's now fully trainable)
        L3          = np.random.normal(0., 0.1)
        L3          = 0 # continue to override this as L3 doesn't train well

        # The qphot mass ratio value (MB/MA) affects the lightcurve via the
        # ellipsoidal effect from the distortion of the stars' shape. Generate
        # a value from the ratio of the radii (or set to -1 to force spherical)
        qphot       = np.random.normal(loc=k, scale=0.1)

        # We generate ecc and omega (argument of periastron) from appropriate
        # distributions and then we subsequently calculate the values of
        # the esinw and ecosw labels and primary/secondary impact params.
        ecc         = np.abs(np.random.normal(loc=0.0, scale=0.2))
        omega       = np.random.uniform(low=0., high=360.) * u.deg

        # Now we can calculate the derived values
        omega_rad   = omega.to(u.rad).value
        esinw       = ecc * np.sin(omega_rad)
        ecosw       = ecc * np.cos(omega_rad)
        rA          = rA_plus_rB / (1 + k)
        rB          = rA_plus_rB - rA
        imp_params  = orbital.impact_parameter(rA, inc, ecc, None, esinw, orbital.EclipseType.BOTH)

        # Create the pset dictionary.
        generated_counter += 1
        pset = {
            "id":           f"{generated_counter:06d}",

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

            **_def_limb_darkening_params,

            # Further params for potential use as labels/features
            "rA":           rA,
            "rB":           rB,
            "e":            ecc,
            "omega":        omega.to(u.deg).value,
            "bP":           imp_params[0],
            "bS":           imp_params[1],                      
        }

        if _is_usable_system(pset):
            yield pset
            usable_counter += 1
            if verbose and usable_counter % 100 == 0:
                print(f"{label}: Generated {usable_counter:,} usable",
                      f"systems from {generated_counter:,} distinct configurations.")


def _calculate_file_splits(instance_count: int, file_count: int) -> List[int]:
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


def _is_usable_system(pset: Dict) -> bool:
    """
    Checks various pset values to decide whether this represents a usable system.
    Checks on;
    - is system physically plausible
    - will it generate eclipses
    - is it suitable for modelling with JKTEBOP
    """
    usable = False
    if pset:
        # Physically plausible
        k = pset.get("k", 0)
        usable = k > 0 and pset.get("J", 0) > 0 \
            and pset.get("qphot", 0) > 0 and pset.get("ecc", 0) < 1

        # Will eclipse
        if usable:
            usable = all(pset.get(b, 0) <= 1 + k for b in ["bP", "bS"])

        # Compatible with JKTEBOP restrictions
        # Soft restriction of rA & rB both < 0.2 as its model is not suited to higher
        # Hard resstrictions of rA+rB<0.8 (covered by above), inc and L3
        if usable:
            usable = all(pset.get(r, 0) <= 0.2 for r in ["rA", "rB"]) \
                and pset.get("inc", 0) > 50 \
                and -1 < pset.get("L3", 0) < 1
    return usable
