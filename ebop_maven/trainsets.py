"""
Functions for generating binary system instances and writing their parameters to trainset csv files.
"""
# We use some ucase variable names as they match the equivalent symbol
# pylint: disable=invalid-name

from typing import Dict, List, Iterable
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
_def_limb_darkening_params = {"LDA": "quad", "LDB": "quad",
                              "LDA1": 0.25,  "LDB1": 0.25,
                              "LDA2": 0.22,  "LDB2": 0.22 }

# The full set of parameters available for histograms, their #bins and plot labels
histogram_params = {
    "rA_plus_rB":   (100, r"$r_{A}+r_{B}$"),
    "k":            (100, r"$k$"),
    "inc":          (100, r"$i~(^{\circ})$"),
    "qphot":        (100, r"$q_{phot}$"),
    #"L3":           (100, r"$L_3$"), # currently always zero
    "e":            (100, r"$e$"),
    "omega":        (100, r"$\omega~(^{\circ})$"),
    "J":            (100, r"$J$"),
    "ecosw":        (100, r"$e\,\cos{\omega}$"),
    "esinw":        (100, r"$e\,\sin{\omega}$"),
    "rA":           (100, r"$r_A$"),
    "rB":           (100, r"$r_B$"),
    "bP":           (100, r"$b_{prim}$")
}

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
The random seed to use for selections:  {seed}\n""")
        if simulate:
            print("Simulate requested so no files will be written.\n")

    if verbose:
        start_time = default_timer()

    # This is reasonably quick so there's no need to use a process pool here.
    # If we decide we need parallel running we'll have to look at setting the seed per file.
    np.random.seed(seed)
    for ix, file_inst_count in enumerate(_calculate_file_splits(instance_count, file_count), 1):
        out_file = output_dir / f"trainset{ix:03d}.csv"
        psets = [*generate_instances_from_distributions(file_inst_count, out_file.stem, verbose)]
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
    generated_counter = 0
    usable_counter = 0
    set_id = label.replace("trainset", "")

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
        imp_prm     = orbital.impact_parameter(rA, inc, ecc, None, esinw, orbital.EclipseType.BOTH)

        # Create the pset dictionary.
        generated_counter += 1
        pset = {
            "id":           f"{set_id}/{generated_counter:06d}",

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
            "bP":           imp_prm[0],
            "bS":           imp_prm[1],                      
        }

        if _is_usable_system(pset):
            yield pset
            usable_counter += 1
            if verbose and usable_counter % 100 == 0:
                print(f"{label}: Generated {usable_counter:,} usable",
                      f"instances from {generated_counter:,} distinct configurations.")


def write_trainset_from_models(pspace_file: Path,
                               output_dir: Path,
                               mission_name: str="TESS",
                               models_name: str="MIST",
                               drop_ratio: float=0.5,
                               seed: float=42,
                               verbose: bool=False,
                               simulate: bool=False) -> None:
    """
    Writes trainset csv file(s) with instances created by using stellar models
    to generate plausible binary systems from every combination in the input
    parameter space.

    :psapce_file: the input parameter space json file
    :output_dir: the directory to write the files to
    :mission_name: name of the observing mission; TESS or Kepler 
    :models_name: name of the stellar models to use; MIST or Parsec
    :drop_ratio: the ratio of generated models to randomly discard
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
The systems' parameter space read from: {pspace_file}
The parameter CSVs will be written to:  {output_dir}
The observation mission to mimic:       {mission_name}
The stellar models to lookup:           {models_name}
The ratio of instances to discard:      {drop_ratio}
The random seed to use when selections: {seed}\n""")
        if simulate:
            print("Simulate requested so no files will be written.\n")

    if verbose:
        start_time = default_timer()
    models = StellarModels.get_instance(models_name)
    mission = Mission.get_instance(mission_name)

    # Read in the parameter space
    with open(pspace_file, "r", encoding="utf8") as f:
        pspace = json.load(f)["pspace"]
    pspace["initial_mass"] *= u.solMass
    pspace["period"] *= u.d
    pspace["inc"] *= u.deg
    pspace["omega"] *= u.deg

    # Start by generating distinct pairs of stars; masses, age and metallicity (Z & Y)
    stellar_psets = []
    for Z in pspace["Z"]:
        Y = models.list_distinct_y_values(z_filter=Z)[0] # Expect 1-to-1 relation between Z & Y
        #print(f"Available initial masses: ", parsec.list_distinct_initial_masses())
        initial_masses = sorted(pspace.get("initial_mass", []), reverse=True)
        countback_MB = pspace.get("count_lower_MB_masses", 0)

        for ix_MA, init_MA in enumerate(initial_masses, start=0):
            # The range of B star masses starts at MA and descends "countback" steps.
            # Then find ages applicable to both; limited by the more massive star.
            init_MBs = initial_masses[ix_MA : ix_MA + max([1, 1 + countback_MB])]
            max_mass = max(init_MA, max(init_MBs)) # Not in this case pylint: disable=nested-min-max
            all_ages = sorted(models.list_distinct_ages(max_mass))
            if max_mass <= 1. * u.solMass: # TODO: models have log10 ages so algo could be refined
                ages = [9.5] * u.dex(u.yr) # ~ 3.5 Gyr
            else:
                ages = [all_ages[int(np.ceil(len(all_ages)*.9))]] * u.dex(u.yr) # tends to later M-S

            if verbose:
                phases = [models.lookup_parameters(Z, Y, init_MA, a, ["PHASE"])[0] for a in ages]
                print(f"{models_name} models for mass {max_mass} cover ages ({min(all_ages):.2f},",
                      f"{max(all_ages):.2f}). Will use: {','.join(f'{a:.2f}' for a in ages)}",
                      f"(at phases: {','.join(f'{p:.2f}' for p in phases)}).")
            stellar_psets += [*product([Z], [Y], [init_MA], init_MBs, ages)]

        if verbose:
            print(f"\nGenerated {len(stellar_psets)} unique pairs of stars. Will now",
                   "write a trainset csv for each pair by varying their configuration.")
        for ix, (z, y, ima, imb, age) in enumerate(stellar_psets, start=1):
            out_file = output_dir / f"trainset{ix:03d}.csv"
            label = out_file.stem
            psets = [*generate_instances_from_models(z, y, ima, imb, age,
                                                     pspace["period"],
                                                     pspace["inc"],
                                                     pspace["ecc"],
                                                     pspace["omega"],
                                                     pspace["L3"],
                                                     mission, models, label, verbose)]
            save_ct = pset_ct = len(psets)
            if drop_ratio and pset_ct:
                # Discard the requested proportion of the systems generated, selected at random
                save_ct = round(pset_ct * (1 - drop_ratio))
                save_ixs = sorted(random.Random(f"{label}/{seed}").sample(range(pset_ct), save_ct))
                if verbose:
                    print(f"Retaining {save_ct} of {pset_ct} instance(s) (drop ratio {drop_ratio})")

            if not simulate:
                if save_ct != pset_ct:
                    cols = psets[0].keys() # Generator, so we need to give _write the field names
                    param_sets.write_param_sets_to_csv(out_file, (psets[i] for i in save_ixs), cols)
                elif save_ct:
                    param_sets.write_param_sets_to_csv(out_file, psets)
                elif verbose:
                    print("Nothing to save for ", out_file)

            if verbose:
                print(f"{'Simulated s' if simulate else 'S'}aving {save_ct} inst(s) to", out_file)

    if verbose:
        print(f"\nFinished. The time taken was {timedelta(0, round(default_timer()-start_time))}.")


@u.quantity_input
def generate_instances_from_models(z: float,
                                   y: float,
                                   init_MA: u.solMass,
                                   init_MB: u.solMass,
                                   age: u.dex(u.yr), # type: ignore
                                   periods: Iterable[u.Quantity[u.d]],
                                   incs: Iterable[u.Quantity[u.deg]],
                                   eccs: Iterable[float],
                                   omegas: Iterable[u.Quantity[u.deg]],
                                   L3s: Iterable[float],
                                   mission: Mission,
                                   models: StellarModels,
                                   label: str,
                                   verbose: bool=True):
    """
    Generates the full set of systems for the fixed parameters Z, Y,
    initial masses and age. The pset is populated by taking the Cartesian
    product of the lists of orbital parameters and third lights supplied.

    :z: the metallicity Z value
    :y: the metallicity Y value
    :initial_MA: the initial mass of the primary star in solMass
    :initial_MB: the initial mass of the secondary star in solMass
    :age: the log10(age) of both stars in dex(yr)
    :periods: list of orbital periods to model
    :incs: list of orbital inclinations to model
    :eccs: list of orbital eccentricities to model
    :omegas: list of orbital arguments of periastron to model
    :L3s: list of third light values to model
    :mission: the observation mission the mimic
    :models: the source of stellar model data
    :label: to include in messages
    :verbose: whether to print verbose progress/diagnostic messages
    :returns: a generator over the dictionaries, one per system
    """
    usable_counter = generated_counter = 0
    set_id = label.replace("trainset", "")

    MA, RA, T_eff_A, LA, loggA = models.lookup_stellar_parameters(z, y, init_MA, age)
    MB, RB, T_eff_B, LB, loggB = models.lookup_stellar_parameters(z, y, init_MB, age)
    k = (RB / RA).value
    q = (MB / MA).value
    light_ratio = (LB / LA).value
    J = mission.expected_brightness_ratio(T_eff_A, T_eff_B)

    # Lookup the LD coeffs. Minimum supported T_eff for pow2 coeffs is 350 0K
    LD_ALGO = "pow2"
    ld_coeffs_A = limb_darkening.lookup_tess_pow2_ld_coeffs(loggA, max(T_eff_A, 3500 * u.K))
    ld_coeffs_B = limb_darkening.lookup_tess_pow2_ld_coeffs(loggB, max(T_eff_B, 3500 * u.K))

    if verbose:
        print(f"""
{label}:
    Star A: M={MA:.3f}, R={RA:.3f}, T_eff={T_eff_A:.1f}, L={LA:.3f}, log(g)={loggA:.3f}, LD({LD_ALGO})={ld_coeffs_A}
    Star B: M={MB:.3f}, R={RB:.3f}, T_eff={T_eff_B:.1f}, L={LB:.3f}, log(g)={loggB:.3f}, LD({LD_ALGO})={ld_coeffs_B}
    system: age={age}, z={z}, y={y}, q={q:.3f}, k={k:.3f}, J={J:.3f}, LB/LA={light_ratio:.3f}""")

    for (P, inc, ecc, omega, L3) in product(periods, incs, eccs, omegas, L3s):
        # No point in varying omega when eccentricity is zero. This logic
        # depends on there being e and omega values of zero in the pspace lists.
        generated_counter += 1
        if ecc != 0. or (ecc == 0. and omega == 0.):
            a = orbital.semi_major_axis(MA, MB, P)
            rA = (RA.to(u.solRad) / a.to(u.solRad)).value
            rB = (RB.to(u.solRad) / a.to(u.solRad)).value
            omega_rad = omega.to(u.rad).value
            esinw = ecc * np.sin(omega_rad)
            ecosw = ecc * np.cos(omega_rad)
            imp_prm = orbital.impact_parameter(rA, inc, ecc, None, esinw, orbital.EclipseType.BOTH)

            pset = {
                "id":           f"{set_id}/{generated_counter:06d}",

                # Basic system params for generating the model light-curve
                # The keys (& those for LD below) are those expected by make_dateset
                "rA_plus_rB":   rA + rB,
                "k":            k,
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
                "rA":           rA,
                "rB":           rB,
                "e":            ecc,
                "omega":        omega.to(u.deg).value,
                "bP":           imp_prm[0],
                "bS":           imp_prm[1],

                # For reference: physical params.
                "LB_over_LA":   light_ratio,
                "P":            P.to(u.d).value,                   
                "a":            a.to(u.solRad).value,
                "RA":           RA.to(u.solRad).value,
                "RB":           RB.to(u.solRad).value,
                "MA":           MA.to(u.solMass).value,
                "MB":           MB.to(u.solMass).value
            }

            if _is_usable_system(pset):
                yield pset
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
