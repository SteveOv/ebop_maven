""" Script for generating the full synthetic testing dataset. """
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
# Put these after the above environ statements so the values are picked up
from traininglib import datasets, plots
from traininglib.mistisochrones import MistIsochrones

from ebop_maven import deb_example
from ebop_maven.libs import orbital, limb_darkening
from ebop_maven.libs.mission import Mission
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

DATASET_SIZE = 20000
RESUME = False
dataset_dir = Path("./datasets/synthetic-mist-tess-dataset/")
dataset_dir.mkdir(parents=True, exist_ok=True)

# TODO: better way to share inst over multiple calls to generate_instances_from_mist_models()
_mist_isochones = MistIsochrones()

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
    # pylint: disable=too-many-locals, too-many-statements, invalid-name
    generated_counter = 0
    usable_counter = 0
    set_id = ''.join(filter(str.isdigit, label))

    # Don't use the built-in hash() function; it's not consistent across processes!!!
    seed = int.from_bytes(hashlib.shake_128(label.encode("utf8")).digest(8))
    rng = np.random.default_rng(seed)

    mission = Mission.get_instance("TESS")

    feh_values = _mist_isochones.list_metallicities()
    min_phase, max_phase = 0.0, 2.0 # M-S to RGB
    min_mass_value, max_mass_value = 0.4, 20.0
    cols = ["star_mass", "log_R", "log_Teff", "log_L", "log_g"]

    while usable_counter < instance_count:
        while True: # imitate "loop and a half" / do ... until logic
            # Get a list of initial masses at a random metallicity & age to choose our stars from
            feh = rng.choice(feh_values)
            ages = _mist_isochones.list_ages(feh, min_phase, max_phase)
            while True:
                age = rng.choice(ages) * u.dex(u.yr)
                init_masses = _mist_isochones.list_initial_masses(feh, age.value,
                                                                  min_phase, max_phase,
                                                                  min_mass_value, max_mass_value)
                if len(init_masses):
                    break

            # First choose the primary mass based on an IMF and multiplicity probability function
            # Choose our stars and then get the basic physical params from the isochrones
            probs = np.power(init_masses, -2.35)        # Salpeter IMF
            probs *= np.tanh(0.31 * init_masses + .18)  # Wells & Prsa Primary multiplicity frac
            probs = np.divide(probs, np.sum(probs))     # Scaled to get a pmf() == 1
            init_MA = rng.choice(init_masses, p=probs) * u.solMass

            init_MB_mask = (init_masses >= min_mass_value) & (init_masses < init_MA.value)
            if any(init_MB_mask):
                init_MB = rng.choice(init_masses[init_MB_mask]) * u.solMass
            else:
                init_MB = init_MA

            results = _mist_isochones.lookup_stellar_params(feh, age.value, init_MA.value, cols)
            MA          = results["star_mass"] * u.solMass
            RA          = np.power(10, results["log_R"]) * u.solRad
            T_eff_A     = np.power(10, results["log_Teff"]) * u.K
            LA          = np.power(10, results["log_L"]) * u.solLum
            loggA       = results["log_g"] * u.dex(u.cm / u.s**2)

            results = _mist_isochones.lookup_stellar_params(feh, age.value, init_MB.value, cols)
            MB          = results["star_mass"] * u.solMass
            RB          = np.power(10, results["log_R"]) * u.solRad
            T_eff_B     = np.power(10, results["log_Teff"]) * u.K
            LB          = np.power(10, results["log_L"]) * u.solLum
            loggB       = results["log_g"] * u.dex(u.cm / u.s**2)

            # Now find the minimum separation, to give the min period, which will be the greater of:
            # . 3(RA+RB) / 2(1-e) (Wells & Prsa) (assuming e==0 for now)
            # . max(5*RA, 5*RB) (based on JKTEBOP recommendation for rA <= 0.2, rB <= 0.2)
            a_min = max(3/2*(RA+RB), 5*RA, 5*RB)
            per_min = orbital.orbital_period(MA, MB, a_min).to(u.d).value

            # We generate period, inc, and omega (argument of periastron) from uniform distributions
            per         = rng.uniform(low=per_min, high=max(per_min, 25)) * u.d
            inc         = rng.uniform(low=50., high=90.00001) * u.deg
            omega       = rng.uniform(low=0., high=360.) * u.deg

            # Eccentricity from uniform distribution, subject to a maximum value which depends on
            # orbital period/seperation (again, based on Wells & Prsa; Moe & Di Stefano)
            a = orbital.semi_major_axis(MA, MB, per)
            if per <= 2 * u.d:
                ecc = 0
            else:
                e_max = max(min(1-(per.value/2)**(-2/3), 1-(1.5*(RA+RB)/a).value), 0)
                ecc = rng.uniform(low=0, high=e_max)

            # We're once more predicting L3 as JKTEBOP is being updated to support
            # negative L3 input values (so it's now fully trainable)
            L3          = rng.normal(0., 0.1)
            L3          = 0 # continue to override this as L3 doesn't train well

            # Now we can calculate other params which we need to decide whether to use this
            q = (MB / MA).value
            rA = (RA.to(u.solRad) / a.to(u.solRad)).value
            rB = (RB.to(u.solRad) / a.to(u.solRad)).value
            inc_rad = inc.to(u.rad).value
            omega_rad = omega.to(u.rad).value
            esinw = ecc * np.sin(omega_rad)
            ecosw = ecc * np.cos(omega_rad)
            imp_prm = orbital.impact_parameter(rA, inc, ecc, None, esinw, orbital.EclipseType.BOTH)

            generated_counter += 1
            inst_id = f"{set_id}/{generated_counter:06d}"
            # Assume J will be OK to defer expensive calc
            # Use eclipse_baseline of 0.75 rather than 1.0 so we don't allow negligible eclipses
            if datasets.is_usable_system(rA, rB, 1.0, q, ecc, inc, imp_prm, eclipse_baseline=0.75):
                break
        # End of do ... until bock

        # Central surface brightness ratio (i.e. in absence of LD) within the mission's bandpass
        J = mission.expected_brightness_ratio(T_eff_A, T_eff_B)

        # Lookup the LD coeffs. Minimum supported T_eff for pow2 coeffs is 3500 K
        LD_ALGO = "pow2"
        ld_coeffs_A = limb_darkening.lookup_tess_pow2_ld_coeffs(loggA, max(T_eff_A, 3500 * u.K))
        ld_coeffs_B = limb_darkening.lookup_tess_pow2_ld_coeffs(loggB, max(T_eff_B, 3500 * u.K))

        # Now we have to decide an appropriate Gaussian noise SNR to apply.
        # Randomly choose an apparent mag in the TESS photometric range then derive
        # the SNR (based on a linear regression fit of Álvarez et al. (2024) Table 2).
        apparent_mag = rng.uniform(6, 18)
        snr = np.add(np.multiply(apparent_mag, -2.32), 59.4)

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

            # Will be used to add Gaussian noise to the light-curve data generated from these params
            "snr":          snr,

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



# ------------------------------------------------------------------------------
# Makes the full synthetic test dataset based on the above generator function
# which generates random plausible dEB systems based on MIST stellar models.
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    with redirect_stdout(Tee(open(dataset_dir/"dataset.log", "w", encoding="utf8"))):
        datasets.make_dataset(instance_count=DATASET_SIZE,
                              file_count=10,
                              output_dir=dataset_dir,
                              generator_func=generate_instances_from_mist_models,
                              file_prefix="trainset",
                              valid_ratio=0.,
                              test_ratio=1.,
                              max_workers=5,
                              save_param_csvs=True,
                              verbose=True,
                              simulate=False)

        # TODO: Update plot_trainset_histograms so that we can change name of the csv/dataset files
        # Histograms are generated from the CSV files (as they cover params not in the dataset)
        plots.plot_trainset_histograms(dataset_dir, dataset_dir/"synth-histogram-full.png", cols=4)
        plots.plot_trainset_histograms(dataset_dir, dataset_dir/"synth-histogram-main.eps", cols=2,
                                    params=["rA_plus_rB", "k", "J", "inc", "ecosw", "esinw"])

        # Simple diagnostic plot of the mags feature of a small sample of the instances.
        dataset_files = sorted(dataset_dir.glob("**/*.tfrecord"))
        ids, _, _, _ = deb_example.read_dataset(dataset_files)
        fig = plots.plot_dataset_instance_mags_features(dataset_files, ids[:30])
        fig.savefig(dataset_dir / "sample.pdf")
