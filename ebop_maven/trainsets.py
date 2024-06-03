"""
Functions for generating binary system instances and writing their parameters to trainset csv files.
"""
# Use ucase variable names where they match the equivalent symbol and pylint can't find units alias
# pylint: disable=invalid-name, no-member

from typing import Dict, List, Iterable, Callable, Generator
from pathlib import Path
from inspect import getsourcefile
from timeit import default_timer
from datetime import timedelta
from itertools import product, zip_longest
import json
import random
import math

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from .libs import param_sets, orbital, limb_darkening
from .libs.mission import Mission
from .libs.stellarmodels import StellarModels

_this_dir = Path(getsourcefile(lambda:0)).parent

# Useable "general use" limb-darkening algo and coefficients
# for a F-type star T_eff~7200 K and logg~4.0
_def_limb_darkening_params = {"LDA": "quad", "LDB": "quad",
                              "LDA1": 0.28,  "LDB1": 0.28,
                              "LDA2": 0.22,  "LDB2": 0.22 }

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

def write_trainset(instance_count: int,
                   file_count: int,
                   output_dir: Path,
                   generator_func: Callable[[int, str, bool], Generator[dict[str, any], any, any]],
                   seed: float=42,
                   verbose: bool=False,
                   simulate: bool=False) -> None:
    """
    Writes trainset csv file(s) with instances created by the chosen generator_func.

    :instance_count: the number of training instances to create
    :file_count: the number of files to spread them over
    :output_dir: the directory to write the files to
    :generator_func: the function to call to generate the required number of systems
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
The random seed to use for selections:  {seed}\n""")
        if simulate:
            print("Simulate requested so no files will be written.\n")

    if verbose:
        start_time = default_timer()

    # This is reasonably quick so there's no need to use a process pool here.
    # If we decide we need parallel running we'll have to look at setting the seed per call out.
    np.random.seed(seed)
    for ix, file_inst_count in enumerate(_calculate_file_splits(instance_count, file_count), 1):
        out_file = output_dir / f"trainset{ix:03d}.csv"
        psets = [*generator_func(file_inst_count, out_file.stem, verbose)]
        if verbose:
            print(f"{'Simulated s' if simulate else 'S'}aving {len(psets)} inst(s) to {out_file}")
        if not simulate:
            param_sets.write_param_sets_to_csv(out_file, psets)

    if verbose:
        print(f"\nFinished. The time taken was {timedelta(0, round(default_timer()-start_time))}.")


def generate_instances_from_distributions(instance_count: int, label: str, verbose: bool=False):
    """
    Generates the requested number of instances based on random distributions
    system of parameters.

    :instance_count: the number of systems to generate
    :verbose: whether to print out verbose progress/diagnostics information
    :returns: a generator over the dictionaries, one per system
    """
    # pylint: disable=too-many-locals
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

            # We're once more predicting L3 as JKTEBOP is being updated to support
            # negative L3 input values (so it's now fully trainable)
            L3          = np.random.normal(0., 0.1)
            L3          = 0 # continue to override this as L3 doesn't train well

            # The qphot mass ratio value (MB/MA) affects the lightcurve via the ellipsoidal effect
            # from the distortion of the stars' shape. Generate a value from the ratio of the radii
            # (or set to -1 to force spherical). Standard homology M-R ratios are a starting point;
            # - low mass M-S stars;     M \propto R^2.5
            # - high-mass M-S stars;    M \propto R^1.25
            qphot       = np.random.normal(loc=k**2, scale=0.1)

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
            if _is_usable_system(rA, rB, J, qphot, ecc, inc, imp_prm):
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

            **_def_limb_darkening_params,

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



def generate_instances_from_mist_models(instance_count: int, label: str, verbose: bool=False):
    """
    Generates the requested number of system instances with a combination of random selecion
    and lookups of MIST stellar models. The following steps are carried out for each candidate
    instance;
    - random selection of Z, initial masses and age from suitable values available in MIST data
    - lookup of current mass, radius, T_effs, log(g) and luminosity values for each component
    - random selection of P, ecc, omega and inc from continuous distributions
    - calculation of rA+rB, k, J, ecosw, esinw, bP and LD params from above
    subject to checks that the system is plausible, will eclipse and is suitable for JKTEBOP

    :instance_count: the number of systems to generate
    :verbose: whether to print out verbose progress/diagnostics information
    :returns: a generator over the dictionaries, one per system
    """
    # pylint: disable=too-many-locals
    generated_counter = 0
    usable_counter = 0
    set_id = label.replace("trainset", "")

    # These are the available values from which we can select MIST lookup values
    models = StellarModels.get_instance("MIST")
    mission = Mission.get_instance("TESS")
    z_values = models.list_distinct_z_values()
    initial_mass_values = models.list_distinct_initial_masses()
    initial_mass_values = [m.value for m in initial_mass_values if 0.4 <= m.value <= 6.4]

    while usable_counter < instance_count:
        while True: # imitate "loop and a half" / do ... until logic
            # These are the basic parameters we need for MIST lookups
            z = np.random.choice(z_values)
            y = np.random.choice(models.list_distinct_y_values(z))
            init_MA = np.random.choice(initial_mass_values) * u.solMass
            init_MB = np.random.choice(initial_mass_values) * u.solMass
            if init_MB > init_MA:
                init_MA, init_MB = init_MB, init_MA

            # TODO: age algo copied over; look to move to random selection. If we implement random
            #       selection we probably want to filter on phase which isn't currently supported.
            age = 9.5 * u.dex(u.yr) # ~ 3.5 Gyr
            max_mass = max(init_MA, init_MB)
            if max_mass > 1.0 * u.solMass:
                all_ages = sorted(models.list_distinct_ages(max_mass))
                age = all_ages[int(np.ceil(len(all_ages) * 0.9))]# tends to later M-S

            # With which we can lookup the current physical stellar parameters
            MA, RA, T_eff_A, LA, loggA = models.lookup_stellar_parameters(z, y, init_MA, age)
            MB, RB, T_eff_B, LB, loggB = models.lookup_stellar_parameters(z, y, init_MB, age)

            # We generate period, inc, ecc and omega (argument of periastron) from
            # appropriate distributions and then we subsequently calculate the values
            # of the esinw and ecosw labels and primary/secondary impact params.
            per         = np.random.uniform(low=2.0, high=25) * u.d
            inc         = np.random.uniform(low=70., high=90.00001) * u.deg
            ecc         = np.abs(np.random.normal(loc=0.0, scale=0.2))
            omega       = np.random.uniform(low=0., high=360.) * u.deg

            # We're once more predicting L3 as JKTEBOP is being updated to support
            # negative L3 input values (so it's now fully trainable)
            L3          = np.random.normal(0., 0.1)
            L3          = 0 # continue to override this as L3 doesn't train well

            # Now we can calculate other params which we need to decide whether to use this
            q = (MB / MA).value
            a = orbital.semi_major_axis(MA, MB, per)
            rA = (RA.to(u.solRad) / a.to(u.solRad)).value
            rB = (RB.to(u.solRad) / a.to(u.solRad)).value
            inc_rad = inc.to(u.rad).value
            omega_rad = omega.to(u.rad).value
            esinw = ecc * np.sin(omega_rad)
            ecosw = ecc * np.cos(omega_rad)
            imp_prm = orbital.impact_parameter(rA, inc, ecc, None, esinw, orbital.EclipseType.BOTH)

            generated_counter += 1
            inst_id = f"{set_id}/{generated_counter:06d}"
            if _is_usable_system(rA, rB, 1.0, q, ecc, inc, imp_prm): # Assume J will be OK
                break
        # End of do ... until bock

        # Central surface brightness ratio (i.e. in absence of LD) within the mission's bandpass
        J = mission.expected_brightness_ratio(T_eff_A, T_eff_B)

        # Lookup the LD coeffs. Minimum supported T_eff for pow2 coeffs is 350 0K
        LD_ALGO = "pow2"
        ld_coeffs_A = limb_darkening.lookup_tess_pow2_ld_coeffs(loggA, max(T_eff_A, 3500 * u.K))
        ld_coeffs_B = limb_darkening.lookup_tess_pow2_ld_coeffs(loggB, max(T_eff_B, 3500 * u.K))

        yield {
            "id":           inst_id,

            # Basic system params for generating the model light-curve
            # The keys (& those for LD below) are those expected by make_dateset
            "rA_plus_rB":   rA + rB,
            "k":            rB / rA,
            "inc":          inc.to(u.deg).value, 
            "qphot":        q,
            "ecosw":        ecosw,
            "esinw":        esinw,
            "J":            J,
            "L3":           L3,

            # System specific LD algo/coeffs for the generation of model light-curve
            "LDA":          LD_ALGO,
            "LDB":          LD_ALGO,
            "LDA1":         ld_coeffs_A[0],
            "LDB1":         ld_coeffs_B[0],
            "LDA2":         ld_coeffs_A[1],
            "LDB2":         ld_coeffs_B[1],

            # Further params for potential use as labels/features
            "sini":         np.sin(inc_rad),
            "cosi":         np.cos(inc_rad),
            "rA":           rA,
            "rB":           rB,
            "ecc":          ecc,
            "omega":        omega.to(u.deg).value,
            "bP":           imp_prm[0],
            "bS":           imp_prm[1],

            # For reference: physical params.
            "LB_over_LA":   (LB / LA).value,
            "P":            per.to(u.d).value,                   
            "a":            a.to(u.solRad).value,
            "RA":           RA.to(u.solRad).value,
            "RB":           RB.to(u.solRad).value,
            "MA":           MA.to(u.solMass).value,
            "MB":           MB.to(u.solMass).value
        }

        usable_counter += 1
        if verbose and usable_counter % 100 == 0:
            print(f"{label}: Generated {usable_counter:,} usable",
                f"instances from {generated_counter:,} distinct configurations.")


def plot_trainset_histograms(trainset_dir: Path,
                             plot_file: Path=None,
                             params: List[str]=None,
                             cols: int=3,
                             verbose: bool=True):
    """
    Saves histogram plots to a single figure on a grid of axes. The params will
    be plotted in the order they are listed, scanning from left to right and down.

    :trainset_dir: the directory containing the trainset csv files
    :plot_file: the directory to save the plots. If none, they're saved with the trainset
    :parameters: the list of parameters to plot, or the full list if None.
    See the histogram_parameters attribute for the full list
    :cols: the width of the axes grid (the rows automatically adjust)
    :verbose: whether to print verbose progress/diagnostic messages
    """
    if not params:
        param_specs = histogram_params
    else:
        param_specs = { p: histogram_params[p] for p in params if p in histogram_params }
    csvs = sorted(trainset_dir.glob("trainset*.csv"))

    if param_specs and csvs:
        size = 3
        rows = math.ceil(len(param_specs) / cols)
        _, axes = plt.subplots(rows, cols, sharey="all",
                               figsize=(cols*size, rows*size), constrained_layout=True)
        if verbose:
            print(f"Plotting histograms in a {cols}x{rows} grid for:", ", ".join(param_specs))

        for (ax, field) in zip_longest(axes.flatten(), param_specs):
            if field:
                bins, label = param_specs[field]
                data = [row.get(field, None) for row in param_sets.read_param_sets_from_csvs(csvs)]
                if verbose:
                    print(f"Plotting histogram for {len(data):,} {field} values.")
                ax.hist(data, bins=bins)
                ax.set_xlabel(label)
                ax.tick_params(axis="both", which="both", direction="in",
                               top=True, bottom=True, left=True, right=True)
            else:
                ax.axis("off") # remove the unused ax

        if verbose:
            print("Saving histogram plot to", plot_file)
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file, dpi=100) # dpi is ignored for vector formats
        plt.close()


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


def _is_usable_system(rA: float, rB: float, J: float, qphot: float,
                      ecc: float, inc: float, imp_params: tuple[float]) -> bool:
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
        usable = all(b <= 1 + k for b in imp_params)

    # Compatible with JKTEBOP restrictions
    # Soft restriction of rA & rB both < 0.2 as its model is not suited to higher
    # Hard restrictions of rA+rB<0.8 (covered by above), inc > 50
    # TODO: will need to extend this for L3 if we start to use non-Zero L3 values
    if usable:
        if isinstance(inc, u.Quantity):
            inc = inc.to(u.deg).value
        usable = rA < 0.2 and rB < 0.2 and inc > 50
    return usable
