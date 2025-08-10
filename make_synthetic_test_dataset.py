""" Script for generating the full synthetic testing dataset. """
import os
import sys
from pathlib import Path
from contextlib import redirect_stdout
import hashlib
from inspect import getsource
import numpy as np

# Tell the libraries where the JKTEBOP executable lives.
# The conda yaml based env sets this but it's not set for venvs.
# pylint: disable=wrong-import-position
if not "JKTEBOP_DIR" in os.environ:
    os.environ["JKTEBOP_DIR"] = "~/jktebop/"

from deblib import limb_darkening, orbital
from deblib.mission import Mission
from deblib.constants import M_sun, R_sun

# Put these after the above environ statements so the values are picked up
from traininglib import datasets, plots
from traininglib.mistisochrones import MistIsochrones
from traininglib.tee import Tee

# Optional filtering/processing on generated LCs before acceptance
SWAP_IF_DEEPER_SECONDARY = True

# max fractional radius: JKTEBOP unsuited to close binaries. As a "rule of thumb" the cut-off is
# at ~0.2. We go further so as to test a model which will be able to predict upto and beyond this.
MAX_FRACTIONAL_R = 0.23

DATASET_SIZE = 20000
FILE_PREFIX = "testset"
dataset_dir = Path("./datasets/synthetic-mist-tess-dataset/")
dataset_dir.mkdir(parents=True, exist_ok=True)

# TODO: better way to share inst over multiple calls to generate_instances_from_mist_models()
_mist_isochones = MistIsochrones()

def generate_instances_from_mist_models(label: str):
    """
    Generates system instances with a combination of random selecion and lookups of MIST stellar
    models. The following steps are carried out for each candidate instance;
    - random selection of Z and age from values available in the MIST data
    - random selection of MA from values compatible with above, subject to probabilities derived
      from an initial mass function and a multiplicity function
    - random selection of MB for masses up to and including MA
    - lookup of current mass, radius, T_effs, log(g) and luminosity values for each component
    - random selection of P, ecc, omega and inc from distributions (subject to restrictions)
    - calculation of rA+rB, k, J, ecosw, esinw and bP label values
    - lookup of pow2 LD params based on stars' logg and Teff values
    - random selection of an apparent snr from within plausible range for TESS observations
    subject to checks that the system is plausible, will eclipse and is suitable for JKTEBOP

    :label: a useful label to use within messages
    :returns: a generator over instance parameter dictionaries, one per system
    """
    # pylint: disable=too-many-locals, too-many-statements, invalid-name
    generated_counter = 0
    set_id = ''.join(filter(str.isdigit, label))

    # Don't use the built-in hash() function; it's not consistent across processes!!!
    seed = int.from_bytes(hashlib.shake_128(label.encode("utf8")).digest(8))
    rng = np.random.default_rng(seed)

    mission = Mission.get_instance("TESS")

    feh_values = _mist_isochones.list_metallicities()
    min_phase, max_phase = 0.0, 2.0     # M-S to RGB
    min_mass = 0.07                     # M_sun. From Wells & Prša (although MIST bottoms at 0.1)
    cols = ["star_mass", "log_R", "log_Teff", "log_L", "log_g"]

    while True: # infinite loop; we will continue to yield new instances until closed
        # Get a list of initial masses at a random metallicity & age to choose our stars from
        feh = rng.choice(feh_values)
        ages = _mist_isochones.list_ages(feh, min_phase, max_phase)
        while True:
            age = rng.choice(ages)
            init_Ms = _mist_isochones.list_initial_masses(feh, age, min_phase, max_phase, min_mass)
            if len(init_Ms):
                break

        # First choose the primary mass based on an IMF and multiplicity probabilities
        # Choose our stars and then get the basic physical params from the isochrones
        probs = chabrier_imf_pmf(init_Ms)
        probs *= wells_prsa_multiplicity_pmf(init_Ms)
        probs = np.divide(probs, np.sum(probs))         # Scaled to get a pmf() == 1
        init_MA = rng.choice(init_Ms, p=probs)          # in units of M_sun

        # Now choose the secondary based on probs from the Moe & di Stefano mass ratio distributions
        init_MB_mask = (init_Ms >= min_mass) & (init_Ms <= init_MA)
        if any(init_MB_mask):
            candidates_MB = init_Ms[init_MB_mask]
            candidates_q = np.divide(candidates_MB, init_MA)
            init_MB = rng.choice(candidates_q, p=md_mass_ratio_pmf(candidates_q, init_MA)) * init_MA
            init_MB = candidates_MB[np.abs(candidates_MB - init_MB).argmin()] # Handle if rounded
        else: # Edge case, where we are at the minimum supported mass
            init_MB = init_MA

        results = _mist_isochones.lookup_stellar_params(feh, age, init_MA, cols)
        MA          = results["star_mass"]              # M_sun
        RA          = np.power(10, results["log_R"])    # R_sun
        T_eff_A     = np.power(10, results["log_Teff"]) # K
        LA          = np.power(10, results["log_L"])    # L_sun
        loggA       = results["log_g"]

        results = _mist_isochones.lookup_stellar_params(feh, age, init_MB, cols)
        MB          = results["star_mass"]              # M_sun
        RB          = np.power(10, results["log_R"])    # R_sun
        T_eff_B     = np.power(10, results["log_Teff"]) # K
        LB          = np.power(10, results["log_L"])    # L_sun
        loggB       = results["log_g"]

        # The minimum period from Kepler's 3rd law based on the minimum supported separation
        min_a = max(RA, RB) / MAX_FRACTIONAL_R          # R_sun
        min_per = orbital.orbital_period(MA * M_sun, MB * M_sun, min_a * R_sun).nominal_value # s
        min_per /= 86400                                # d

        # We generate period, cos(i), and omega (argument of periastron) from uniform distributions
        per         = rng.uniform(low=min_per, high=max(min_per, 25))   # d
        cosi        = rng.uniform(high=0.643, low=0.)   # uniform in cosi between ~50 & 90 deg
        inc_rad     = np.arccos(cosi)
        inc         = np.degrees(inc_rad)               # deg
        omega       = rng.uniform(low=0., high=360.)    # deg

        # Eccentricity from uniform distribution, subject to a maximum value which depends on
        # orbital period/seperation (again, based on Wells & Prsa (eqn 6); Moe & Di Stefano)
        a = (orbital.semi_major_axis(MA * M_sun, MB * M_sun, per * 86400) / R_sun).nominal_value
        if per <= 2:
            ecc = 0
        else:
            e_max = max(min(1-(per/2)**(-2/3), 1-(1.5*(RA+RB)/a)), 0)
            ecc = rng.uniform(low=0, high=e_max)

        # Introduce some third light (versions of JKTEBOP up to 43 do not accept negative L3 input)
        L3          = max(0, rng.normal(loc=0.0, scale=0.05)) # ~half will be zero

        # Now we can calculate other params which we need to decide whether to use this
        q = MB / MA
        rA = RA / a
        rB = RB / a
        omega_rad = np.radians(omega)
        esinw = ecc * np.sin(omega_rad)
        ecosw = ecc * np.cos(omega_rad)
        imp_prm = (
            orbital.impact_parameter(rA, inc, ecc, esinw, False),
            orbital.impact_parameter(rA, inc, ecc, esinw, True)
        )

        # Check usability before calculating J & LD params to avoid expensive calls if unnecessary.
        generated_counter += 1
        if is_usable_instance(rA+rB, rB/rA, 1., q, ecc, L3, imp_prm[0], imp_prm[1], rA, rB, inc):
            # Central surface brightness ratio (i.e. in absence of LD) within the mission's bandpass
            J = mission.expected_brightness_ratio(T_eff_A, T_eff_B)

            # Lookup the nearest matching LD coeffs
            LD_ALGO = "pow2"
            ld_coeffs_A = limb_darkening.lookup_pow2_coefficients(loggA, T_eff_A, "TESS")
            ld_coeffs_B = limb_darkening.lookup_pow2_coefficients(loggB, T_eff_B, "TESS")

            yield {
                "id":           f"{set_id}/{generated_counter:06d}",

                # Basic system params for generating the model light-curve
                # The keys (& those for LD below) are those expected by make_dateset
                "rA_plus_rB":   rA + rB,
                "k":            rB / rA,
                "inc":          inc, 
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

                # By specifying an apparent mag (within the TESS mission's operating range) we
                # indicate the conditions from which we decide the amount of noise to add.
                # The phase and vertical shifts to the mags data mimic imperfect pre-processing.
                "noise_sigma":  calculate_tess_noise_sigma(apparent_mag=rng.uniform(6, 18)),
                "phase_shift":  rng.normal(0, scale=0.03),
                "mag_shift":    rng.normal(0, scale=0.01),

                # Further params for potential use as labels/features
                "sini":         np.sin(inc_rad),
                "cosi":         cosi,
                "rA":           rA,
                "rB":           rB,
                "ecc":          ecc,
                "omega":        omega,
                "bP":           imp_prm[0],
                "bS":           imp_prm[1],
                "phiS":         orbital.phase_of_secondary_eclipse(ecosw, ecc),
                "dS_over_dP":   orbital.ratio_of_eclipse_duration(esinw),

                # For reference: physical params.
                "LB_over_LA":   LB / LA,
                "P":            per,                   
                "a":            a,
                "RA":           RA,
                "RB":           RB,
                "MA":           MA,
                "MB":           MB
            }

def salpeter_imf_pmf(masses):
    """ Salpeter IMF as a simple power law. """
    # Ignore the normalization constant as this will be normalized later
    imf = np.power(masses, -2.35)
    return imf / np.sum(imf)

def chabrier_imf_pmf(masses):
    """ 
    Chabrier (2003PASP..115..763C) two part IMF (Table I)
    as summarized by Maschberger (2013MNRAS.429.1725M)
    """
    imf = np.zeros_like(masses, dtype=float)

    # p(M) = 0.0443 * M^-2.3 for M >= 1.0 MSun
    mask = masses >= 1.0
    coeffs =  0.0443
    imf[mask] = np.multiply(coeffs, np.power(masses[mask], -2.3))

    # p(M) = 0.158/M * exp[-0.5*(log(M)-log(0.079)/0.69)^2] for M < 1.0 MSun
    mask = ~mask
    coeffs = np.divide(0.158, masses[mask])
    exponent = np.square(np.divide(np.subtract(np.log10(masses[mask]), np.log10(0.079)), 0.69))
    imf[mask] = np.multiply(coeffs, np.exp(np.multiply(-0.5, exponent)))
    return imf / np.sum(imf)

def wells_prsa_multiplicity_pmf(masses):
    """
    Wells & Prša (2021ApJS..253...32W) eqn. 2 with coeff noms as given in the following paragraph
    (to fit data from Duchêne & Kraus [2013ARA&A..51..269D] & Raghavan+ [2010ApJS..190....1R]).
    Based on Arenou (2011AIPC.1346..107A) eqn. 1 with modified coefficients.
    """
    pmf = np.tanh(np.add(np.multiply(0.31, masses), 0.18))
    return pmf / np.sum(pmf)

def md_mass_ratio_pmf(q, mass_prim):
    """
    A q probability distribution, based on the Moe & Di Stefano (2017ApJS..230...15M)
    mass-ratio probability distribution with parameters from Table 13 within the log(P) = 1 regime.
    """
    if isinstance(q, float|int):
        q = np.array([q], float)
    num_probs = len(q)

    if num_probs == 1:
        return np.array([1])

    probs = np.zeros(shape=(num_probs, ), dtype=float)
    (gamma_smallq, gamma_largeq, f_twin) = (0.3, -0.5, 0.30) if mass_prim <= 1.2 \
                                      else (0.2, -0.5, 0.22) if mass_prim <= 5.0 \
                                      else (0.1, -0.5, 0.17) if mass_prim <= 9.0 \
                                      else (0.1, -0.5, 0.14) if mass_prim <= 16.0 \
                                      else (0.1, -0.5, 0.08)

    smallq_mask = q <= 0.3
    probs[smallq_mask] = np.power(q[smallq_mask], gamma_smallq)

    largeq_mask = ~smallq_mask
    probs[largeq_mask] = np.power(q[largeq_mask], gamma_largeq)

    if f_twin:
        f_twin_mask = q > 0.95
        if any(f_twin_mask):
            # Uniformly trim the f_twin excess fraction across the whole large q range
            auc_excess = np.sum(probs[largeq_mask]) * f_twin
            probs[largeq_mask] -= auc_excess / (num_probs * 0.7)

            # Now uniformly boost the f_twin range by the previously trimmed excess fraction
            probs[f_twin_mask] += auc_excess / (num_probs * 0.05)

    # Ensure the pdf is continuous from smallq to largeq regimes
    if any(smallq_mask):
        probs[smallq_mask] /= probs[smallq_mask][-1] / probs[largeq_mask][0]
    return probs / np.sum(probs)

def calculate_tess_noise_sigma(apparent_mag: float) -> float:
    """
    Calculates the overall random noise sigma for normalized TESS photometric timeseries fluxes
    for a target with the passed apparent mag.
    """
    # These are coeffs in Stassun+2018 (TESS Input Catalog & Candidate Target List) [V1]
    # https://arxiv.org/abs/1706.00495v1 on page 24, however they're replaced with a ref to a
    # Pepper+2018 paper (in prep) in later versions which I have yet to find.
    # Ultimately the coeffs are derived from Fig 8 of Ricker+(2015JATIS...1a4003R) *The TESS paper*
    # _ln_noise_sigma_poly = np.poly1d([4.73508403525e-5, -0.0022308015894, 0.0395908321369,
    #                                   -0.285041632435, 0.850021465753, 3.29685004771])
    # noise_sigma_ppm_hr = np.exp(_ln_noise_sigma_poly)

    # This is my exp trend fit to datapoints on Ricker+(2015JATIS...1a4003R) Fig 8. Using 60 as the
    # noise floor, as given in the narrative, and made pessimistic by fitting to 10^5.5 at mag 18.
    noise_sigma_ppm_hr = max(60, 0.402 * np.exp(0.665 * apparent_mag))

    # Change (sigma) from ppm/hr to ppm/2min then undo the ppm
    return noise_sigma_ppm_hr * (2/60)**0.5 / 10**6

def is_usable_instance(rA_plus_rB: float, k: float, J: float, qphot: float, ecc: float,
                       L3: float, bP: float, bS: float, rA: float, rB: float, inc: float,
                       phiP: float=0, phiS: float=0.5, depthP: float=100, depthS: float=100,
                       **_ # Used to ignore any unexpected **params
                       ) -> bool:
    """
    Checks various parameter values to decide whether this represents a usable system.
    Checks on;
    - is system physically plausible
    - will it generate eclipses and are they sufficiently prominent to be usable
    - is it suitable for modelling with JKTEBOP
    """
    # pylint: disable=invalid-name, too-many-arguments, unused-argument
    # Physically plausible
    usable = 0 <= ecc < 1 and -1 < L3 < 1

    # Will eclipse, and that they are sufficiently prominent to be useful for testing
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
# Makes the full synthetic test dataset based on the above generator function
# which generates random plausible dEB systems based on MIST stellar models.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if (dataset_dir/"dataset.log").exists():
        response =  input("\nFiles exist for this dataset. Continue and overwrite y/N? ")
        if response.strip().lower() not in ["y", "yes"]:
            sys.exit()

    with redirect_stdout(Tee(open(dataset_dir/"dataset.log", "w", encoding="utf8"))):

        code_file = dataset_dir / "parameter-distributions.txt"
        with code_file.open("w", encoding="utf8") as of:
            of.write(f"SWAP_IF_DEEPER_SECONDARY = {SWAP_IF_DEEPER_SECONDARY}\n")
            of.write(f"MAX_FRACTIONAL_R = {MAX_FRACTIONAL_R}\n\n")
            of.write(getsource(generate_instances_from_mist_models))
            of.write("\n\n")
            of.write(getsource(is_usable_instance))
        print(f"Saved copies of the param distribution & constraint functions to {code_file.name}")

        datasets.make_dataset(instance_count=DATASET_SIZE,
                              file_count=10,
                              output_dir=dataset_dir,
                              generator_func=generate_instances_from_mist_models,
                              check_func=is_usable_instance,
                              swap_if_deeper_secondary=SWAP_IF_DEEPER_SECONDARY,
                              file_prefix=FILE_PREFIX,
                              valid_ratio=0.,
                              test_ratio=1.,
                              max_workers=5,
                              save_param_csvs=True,
                              verbose=True,
                              simulate=False)

        # Histograms are generated from the CSV files as they cover params not saved to tfrecord
        csvs = sorted(dataset_dir.glob(f"**/{FILE_PREFIX}*.csv"))
        plots.plot_dataset_histograms(csvs, cols=5).savefig(dataset_dir/"synth-histogram-full.png")
        plots.plot_dataset_histograms(csvs, ["rA_plus_rB", "k", "J", "inc", "ecosw", "esinw"],
                                      cols=2).savefig(dataset_dir/"synth-histogram-main.pdf")

        # Simple diagnostic plot of the mags feature of a small sample of the instances.
        for dataset_file in sorted(dataset_dir.glob(f"**/{FILE_PREFIX}000.tfrecord")):
            print(f"Plotting a sample of the {dataset_file.parent.name} subset's mags features")
            fig = plots.plot_dataset_instance_mags_features([dataset_file], mags_wrap_phase=0.5,
                                                            cols=5, max_instances=50)
            fig.savefig(dataset_dir / f"sample-{dataset_file.parent.name}.png", dpi=150)
            fig.clf()
