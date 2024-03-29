""" Utility functions for Light-curves. """
from typing import Union, List, Iterable, Tuple
from pathlib import Path
import math

import numpy as np
from scipy import signal, interpolate
import astropy.units as u
from astropy.io import fits
from astropy.time import Time

import lightkurve as lk
from lightkurve import LightCurve, FoldedLightCurve, LightCurveCollection

# This pylint disable overlooks the issue it has with astropy const aliases
# pylint: disable=no-member

def find_lightcurves(target: any,
                     download_dir: Path,
                     sectors: Union[int, List[int]]=None,
                     mission: str="TESS",
                     author: str="SPOC",
                     exptime: Union[str, float]=None,
                     flux_column: str="sap_flux",
                     quality_bitmask: Union[str, int]="default",
                     force_mast: bool=False,
                     verbose: bool=True) -> LightCurveCollection:
    """
    This performs both a search and download on Lightkurve assets. However,
    if override is False it will firstly try to service the request from assets
    already found in the download_dir to avoid making potentially costly remote
    calls to MAST. The arguments are the most used arguments associated with
    the Lightkurve search_lightcurve() and results.download_all() functions.

    Values are required for sectors, mission and auther in order to try
    servicing the request from the download_dir. The assumption is made that
    the download dir is exclusive to this target as we cannot tie all potential
    values of the target search criterion directly to the contents of fits.
    
    :target: the target search term
    :download_dir: the local directory under which the assets are/will be stored
    :sectors: the sector or sectors search term
    :mission: the mission search term
    :author: the author search term
    :exptime: the exptime (exposure time) search term. Either None, text
    ("long", "short" or "fast") or a numeric value in secords
    :flux_column: the flux column to select when loading the assets
    :quality_bitmask: the mask to apply to the Quality col when loading the assets.
    either text ("default", "hard" or "hardest") or a numeric bit mask
    :force_mast: if True will always bypass local files and search/download from MAST
    :verbose: if True will output some diagnostics text to the console
    """
    # pylint: disable=too-many-locals, unnecessary-lambda-assignment
    lcs = None
    if mission.lower() not in ["tess", "hlsp"]:
        raise ValueError("only missions TESS and HLSP are currently supported")
    if author.lower() not in ["spoc", "tess-spoc"]:
        raise ValueError("only authors SPOC and TESS-SPOC are currently supported")

    if sectors and not isinstance(sectors, Iterable):
        sectors = [sectors]

    if verbose:
        print(f"Searching for lightcurves based on; target={target}, sectors={sectors},",
              f"mission={mission}, author={author} and exptime={exptime}")

    if not force_mast and sectors and mission and author:
        # We can only reliably shortcut if we have been given a full list criteria.
        if verbose:
            print("Looking for previously downloaded fits within", download_dir)
        # Known filename patterns for author on TESS. The mission doesn't seem to matter:
        # TESS-SPOC: hlsp_tess-spoc_tess_phot_{tic_number:016d}_s{sector:04d}_tess_v1_lc.fits
        # SPOC:      tess{date:13d}_s{sector:04d}_{tic_number:016d}_{sequence??:04d}_lc.fits
        if author.lower() == "spoc":
            glob_pattern = "**/tess*_lc.fits"
        else:
            glob_pattern = "**/hlsp_tess-spoc_*_lc.fits"
        fits_files = [*download_dir.glob(glob_pattern)]

        if fits_files:
            # We have local files that match the file pattern for mission/author
            if verbose:
                print(f"Found {len(fits_files)} existing fits matching mission and author criteria")

            # select only those that match the sector & exptime
            if not exptime:
                check_exptime = None
            else:
                # This is what lambdas are for despite what pylint keeps wittering on about!
                if not isinstance(exptime, str):
                    check_exptime = lambda fits_exptime: fits_exptime == exptime
                elif exptime.lower() == "short":
                    check_exptime = lambda fits_exptime: 60 <= fits_exptime <= 120
                elif exptime.lower() == "fast":
                    check_exptime = lambda fits_exptime: fits_exptime < 60
                else:
                    check_exptime = lambda fits_exptime: fits_exptime > 120

            hduls = [
                h for h in (fits.open(ff) for ff in fits_files)
                    if h[0].header["SECTOR"] in sectors and
                        (not check_exptime
                            or check_exptime(h[1].header["FRAMETIM"] * h[1].header["NUM_FRM"]))
            ]

            if len(hduls) == len(sectors):
                # We're on! Load these into a collection and return them.
                if verbose:
                    print(f"Found the required {len(hduls)} fits also meeting the sectors &",
                          "exptime criteria. Will load the requested lightcurves from these.")
                lcs = LightCurveCollection(
                    lk.read(h.filename(), flux_column=flux_column, quality_bitmask=quality_bitmask)
                        for h in sorted(hduls, key=lambda hdul: hdul[0].header["SECTOR"])
                )

    if not lcs:
        if verbose:
            print("Performing a MAST query based on the criteria.")
        # We've not been able to service the request from already dl assets: usual search & download
        results = lk.search_lightcurve(target, sector=sectors, mission=mission,
                                       author=author, exptime=exptime)
        if verbose:
            print("Criteria met by a MAST search yielding a", results)

        # Work around a recent issues with DL assets - suggested by Pierre Maxted
        if "dataUrl" not in results.table.colnames:
            results.table["dataURL"] = results.table["dataURI"]
        download_dir.mkdir(parents=True, exist_ok=True)
        lcs = results.download_all(quality_bitmask, f"{download_dir}",
                                   flux_column=flux_column, cache=True)
    return lcs


def expected_ratio_of_eclipse_duration(esinw: float) -> float:
    """
    Calculates the expected ratio of eclipse durations dS/dP from e*sin(omega)

    Uses eqn 5.69 from Hilditch (2001) An Introduction to Close Binary Stars
    reworked in terms of dS/dP
    """
    return (esinw + 1)/(1 - esinw)


def expected_secondary_phase(ecosw: float, ecc: float) -> float:
    """
    Calculates the expected secondary (normalized) phase from e*cos(omega) and e

    Uses eqn 5.67 and 5.68 from Hilditch, setting P=1 (normalized) & t_pri=0
    to give phi_sec = t_sec = (X-sinX)/2pi where X=pi+2*atan(ecosw/sqrt(1-e^2))
    """
    x = math.pi + (2*math.atan(ecosw/np.sqrt(1-np.power(ecc, 2))))
    return (x - math.sin(x)) / (2 * math.pi)


def apply_quality_masks(lc: LightCurve,
                        mask_time_ranges: List[Tuple[Time, Time]] = None,
                        verbose: bool = False) -> LightCurve:
    """
    Will return a copy of the passed LightCurve having first masked
    any fluxes whose value is NaN or less than Zero, then having masked
    any requested time ranges.

    :lc: the LightCurve to mask
    :mask_time_ranges: zero or more (from, to) Time ranges to mask out
    :verbose: whether or not to generate progress messages
    :returns: the masked LightCurve
    """
    if not isinstance(lc, LightCurve):
        raise TypeError(f"lc is {type(lc)}. Expected a LightCurve.")

    mask = (np.isnan(lc.flux)) | (lc.flux < 0)
    count_nan_masked = sum(mask.unmasked)
    if verbose:
        print(f"NaN/negative flux masks match {count_nan_masked} row(s).")

    if mask_time_ranges is not None:
        if not isinstance(mask_time_ranges, Iterable):
            raise TypeError(f"mask_time_ranges is {type(mask_time_ranges)}. Expected an Iterable.")

        count_time_masked = 0
        for tr in mask_time_ranges:
            mask |= mask_from_time_range(lc, tr)
        if verbose:
            count_time_masked = sum(mask.unmasked) - count_nan_masked
            print(f"Time range mask(s) matched a further {count_time_masked} row(s).")

    return lc[~mask]


def mask_from_time_range(lc: LightCurve,
                         time_range: Union[Tuple[Time, Time], Time]) \
                            -> List[bool]:
    """
    Creates a mask applicable to the LightCurve over the indicated time range,
    with the mask being applied to all datapoints within the range inclusive.

    :lc: the LightCurve to create the mask for
    :time_range: the (from, to) time range to mask
    """
    if not isinstance(time_range, (Iterable, Time)):
        raise TypeError(f"time_range is {type(time_range)}. Expected an Iterable or Time.")
    if not isinstance(lc, LightCurve):
        raise TypeError(f"lc is {type(lc)}. Expected a LightCurve.")

    time_range = to_lc_time(time_range, lc)
    return (lc.time >= time_range.min()) & (lc.time <= time_range.max())


def to_lc_time(value: Union[Time, np.double, Tuple[np.double], List[np.double,]],
            lc: LightCurve) \
                -> Time:
    """
    Converts the passed numeric value to an astropy Time.
    The magnitude of the time will be used to interpret the format to match LC.

    :value: the value to be converted
    :lc: the light-curve to match format with
    """
    if isinstance(value, Time):
        if value.format == lc.time.format and value.scale == lc.time.scale:
            return value
        raise ValueError("Value's time format/scale does not match the Lightcurve's")

    if isinstance(value, Iterable):
        return Time([to_lc_time(v, lc) for v in value])

    # Otherwise try to match the time format and scale to the Lightcurve
    if value < 4e4:
        if lc.time.format == "btjd":
            return Time(value, format="btjd", scale=lc.time.scale)
        raise ValueError(f"Unexpected value/format ({value}/{lc.time.format}) combination.")
    else:
        if value < 2.4e6:
            value += 2.4e6
        return Time(value, format="jd", scale=lc.time.scale)


def bin_lightcurve(lc: LightCurve,
                   time_bin_seconds: Time,
                   verbose: bool = False) -> LightCurve:
    """
    Will return a binned copy of the passed LightCurve

    :lc: the LightCurve to bin
    :time_bin_seconds: the length of the revised bins
    :verbose: whether or not to generate progress messages
    :returns: the binned LightCurve
    """
    if isinstance(time_bin_seconds, (float, np.double, int)):
        time_bin_seconds *= u.s

    # Can only re-bin to longer than the existing bins
    int_time = (lc.meta["INT_TIME"] + lc.meta["READTIME"]) * u.min
    if time_bin_seconds > int_time:
        orig_len = len(lc)
        lc = lc.bin(time_bin_size=time_bin_seconds, aggregate_func=np.nanmean)
        lc = lc[~np.isnan(lc.flux)] # Binning may have re-introduced NaNs
        if verbose:
            print(f"After binning light-curve has reduced from {orig_len} to {len(lc)} rows.")
    elif verbose:
        print(f"Light-curve already in bins >= {time_bin_seconds}. Will leave unchanged.")
    return lc


def append_magnitude_columns(lc: LightCurve,
                             name: str = "delta_mag",
                             err_name: str = "delta_mag_err"):
    """
    This will append a relative magnitude and corresponding error column
    to the passed LightCurve based on the values in the flux column,

    :lc: the LightCurve to update
    :name: the name of the new magnitude column
    :err_name: the name of the corresponding magnitude error column
    """
    lc[name] = u.Quantity(-2.5 * np.log10(lc.flux.value) * u.mag)
    lc[err_name] = u.Quantity(
        2.5
        * 0.5
        * np.abs(
            np.subtract(
                np.log10(np.add(lc.flux.value, lc.flux_err.value)),
                np.log10(np.subtract(lc.flux.value, lc.flux_err.value))
            )
        )
        * u.mag)


# pylint: disable=too-many-arguments, too-many-locals
def fit_polynomial(times: Time,
                   ydata: u.Quantity,
                   degree: int = 2,
                   iterations: int = 2,
                   res_sigma_clip: float = 1.,
                   reset_const_coeff: bool = False,
                   include_coeffs: bool = False,
                   verbose: bool = False) \
                    -> Union[u.Quantity, Tuple[u.Quantity, List]]:
    """
    Will calculate a polynomial fit over the requested time range and y-data
    values. The fit is iterative; after each iteration the residuals are
    evaluated against a threshold defined by the StdDev of the residuals
    multiplied by res_sigma_clip; any datapoints with residuals greater than
    this are excluded from subsequent iterations.  This approach will exclude
    large y-data excursions, such as eclipses, from influencing the final fit.

    :times: the times (x data)
    :ydata: data to fit to
    :degree: degree of polynomial to fit.  Defaults to 2.
    :iterations: number of fit iterations to run.
    :res_sigma_clip: the factor applied to the residual StdDev to calculate
    the clipping threshold for each new iteration.
    :reset_const_coeff: set True to reset the const coeff to 0 before final fit
    :include_coeffs: set True to return the coefficients with the fitted ydata
    :returns: fitted y data and optionally the coefficients used to generate it.
    """
    pivot_ix = int(np.floor(len(times) / 2))
    pivot_jd = times[pivot_ix].jd
    time_values = times.jd - pivot_jd

    fit_mask = [True] * len(ydata)
    for remaining_iterations in np.arange(iterations, 0, -1):
        # Fit a polynomial to the masked data so that we find its coefficients.
        # For the first iteration this will be all the data.
        coeffs = np.polynomial.polynomial.polyfit(time_values[fit_mask],
                                                  ydata.value[fit_mask],
                                                  deg=degree,
                                                  full=False)

        if remaining_iterations > 1:
            # Find and mask out those datapoints where the residual to the
            # above poly lies outside the requested sigma clip. This stops
            # large excursions, such as eclipses, from influencing the poly fit.
            poly_func = np.polynomial.Polynomial(coeffs)
            fit_ydata = poly_func(time_values)
            resids = ydata.value - fit_ydata
            fit_mask &= (np.abs(resids) <= (np.std(resids)*res_sigma_clip))
        else:
            # Last iteration we generate the poly's y-axis datapoints for return
            if reset_const_coeff:
                if verbose:
                    print("\tResetting const/0th coefficient to zero on request.")
                coeffs[0] = 0
            poly_func = np.polynomial.Polynomial(coeffs)
            fit_ydata = poly_func(time_values) * ydata.unit
            if verbose:
                c_list = ", ".join(f'c{ix} = {c}' for ix, c in enumerate(poly_func.coef))
                print(f"\tGenerated polynomial; y = poly(x, {c_list})",
                      f"(sigma(fit_ydata)={np.std(fit_ydata):.6e})")

    return (fit_ydata, coeffs) if include_coeffs else fit_ydata


def get_reduced_folded_lc(flc: FoldedLightCurve,
                          num_bins: int = 1024,
                          phase_pivot: Union[u.Quantity, float]=0.75,
                          flc_rollover: int = 200,
                          smooth_fit: bool = False,
                          interp_kind: str="linear",) \
                                -> Tuple[u.Quantity, u.Quantity]:
    """
    A data reduction function which gets a reduced set of phase folded 
    delta magnitude data in equal size bins of the requested number. 
    
    The data is sourced by sampling the passed FoldedLightCurve.  In case this 
    does not extend over a complete phase, rows are copied over from opposite
    ends of the phase space data to extend the coverage.  The number of rows 
    copied is controlled by the flc_rollover argument.

    :flc: the source FoldedLightCurve
    :num_bins: the number of equally spaced rows to return
    :phase_pivot: the pivot point about which the fold phase was wrapped to < 0.
    :flc_rollover: the number of row to extend the ends of the source phases by
    :smooth_fit: whether to apply SavGol smoothing while reducing the fold
    :interp_kind: the 1d interpolation algorithm to use when reducing the fold
    :returns: a tuple with requested number or phases and delta magnitudes
    """
    source_phases = np.concatenate([
        flc.phase[-flc_rollover:] -1.,
        flc.phase,
        flc.phase[:flc_rollover] +1.
    ])

    source_delta_mags = np.concatenate([
        flc["delta_mag"][-flc_rollover:],
        flc["delta_mag"],
        flc["delta_mag"][:flc_rollover]
    ])

    # If there is a phase wrap then phases above the pivot will have been
    # wrapped around to <0. Work out what the expected minimum phase should be.
    # Also, ensure we don't try to interpolate beyond the start of the data!
    min_phase = u.Quantity(0)
    if phase_pivot is not None:
        if isinstance(phase_pivot, u.Quantity) and phase_pivot.value:
            min_phase = phase_pivot.value - 1
        elif phase_pivot:
            min_phase = u.Quantity(phase_pivot - 1)
    min_phase = max(min_phase, source_phases.min())

    interp = interpolate.interp1d(source_phases, source_delta_mags, kind=interp_kind)
    reduced_phases = np.linspace(min_phase, min_phase + 1., num_bins + 1)[:-1]
    reduced_mags = interp(reduced_phases)

    # TODO: may remove this as it doesn't particularly help predictions
    if smooth_fit:
        # Apply smoothing to the output. Keep the window and order low otherwise
        # we get artefacts (extra peaks) at the transition in/out of eclipses.
        reduced_mags = signal.savgol_filter(reduced_mags, 7, 3)

    return (reduced_phases, reduced_mags)
