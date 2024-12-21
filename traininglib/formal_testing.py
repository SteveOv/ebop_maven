"""
Helper functions to support interacting with formal test dataset.
Functions in this module know about the format of the formal test dataset config.
"""
# Use ucase variable names where they match the equivalent symbol and pylint can't find units alias
# pylint: disable=invalid-name, no-member, too-many-arguments, too-many-locals
from typing import Dict, List, Tuple, Generator, Union
from pathlib import Path
import json
import re

import numpy as np
import astropy.units as u
from lightkurve import LightCurve, LightCurveCollection
from uncertainties import ufloat, UFloat

from . import pipeline


def iterate_target_configs(target_configs: Union[Path, dict[str, any]],
                           chosen_targets: List[str]=None,
                           include_excluded: bool=False) \
                                -> Generator[Tuple[str, Dict[str, any]], any, None]:
    """
    Sets up a Generator over the target configs within the passed config file.

    :target_configs: the source configuration, either a dict or a Path to a json file
    :chosen_targets: which targets and the expected order, or all in the order held if None
    :include_excluded: whether to include those targets marked as excluded
    :returns: Generator over the chosen targets yielding (str, dict) of the target name and config
    """
    if isinstance(target_configs, dict):
        configs = target_configs
    else:
        with open(target_configs, mode="r", encoding="utf8") as f:
            configs = json.load(f)

    # Only targets we know and order of supplied list (if given) overrides order in config file
    if chosen_targets is None or len(chosen_targets) == 0:
        chosen_targets = list(configs.keys())
    else:
        chosen_targets = [k for k in chosen_targets if k in configs]

    for k in chosen_targets:
        v = configs[k]
        if include_excluded or not v.get("exclude", False):
            yield k, v


def get_labels_for_targets(target_config: Union[Path, dict[str, any]],
                           chosen_labels: list[str],
                           chosen_targets: List[str]=None,
                           include_excluded: bool=False) -> np.ndarray[UFloat]:
    """
    Gets the label values and uncertainties for the chosen set of targets and labels.

    :target_configs: the source configuration, either a dict or a Path to a json file
    :chosen_labels: the labels to get, in the order required
    :chosen_targets: which targets and the expected order, or all in the order held if None
    :include_excluded: whether to include those targets marked as excluded
    :returns: a structured NDArray[UFloat] with columns for the chosen labels and rows
    in the order of the specified targets
    """
    dtype = [(l, np.dtype(UFloat.dtype)) for l in chosen_labels]
    rows = []
    for _, config in iterate_target_configs(target_config, chosen_targets, include_excluded):
        lbl_cfg = config.get("labels", { })
        rows += [tuple(
            ufloat(lbl_cfg.get(l, 0), lbl_cfg.get(f"{l}_err", None) or 0) for l in chosen_labels
        )]
    return np.array(rows, dtype=dtype)


def list_sectors_in_target_config(target_cfg: Dict[str, any]) -> List[int]:
    """
    Returns a list of the sectors configured for this target.

    :target_cfg: the target's config dictionary, as read from the config json
    :returns: a list[int] of the configured sector numbers
    """
    return [int(s) for s in target_cfg["sectors"].keys() if s.isdigit()]


def sector_config_from_target(sector: int, target_cfg: Dict[str, any]) -> Dict[str, any]:
    """
    Get the sector specific config from the passed target config. The sector config
    is the target config coalesced with the sector config so the sector provides overrides.

    :sector: the chosen sector number
    :target_cfg: the target's config dictionary, as read from the config json
    :returns: the sector's configuration dictionary
    """
    sector_cfg = {
        **target_cfg.copy(),
        **target_cfg["sectors"][f"{sector}"]
    }
    sector_cfg.pop("sectors", None)
    return sector_cfg


def prepare_lightcurve_for_target(target: str,
                                  target_cfg: Dict[str, any],
                                  verbose: bool=True) \
                                    -> Tuple[LightCurve, int]:
    """
    Will prepare a light curve for the passed test target.
    If the target has multiple sectors configured, these will be downloaded
    separately and stitched into a single Lightcurve

    :target: the name/search term for the target
    :target_cfg: the target's config dictionary, as read from the config json
    :verbose: whether to write detailed progress information to stdout
    :returns: a tuple of the combined LightCurve for the configured sectors and the sector count
    """
    # Handle using a different search term than the target name
    search_term = target_cfg.get("search_term", target) or target
    sectors = list_sectors_in_target_config(target_cfg)
    if verbose:
        print(f"Downloading and stitching the light curves for {target}",
              f"(search term='{search_term}')" if search_term != target else "",
              "sector(s)", 
              ", ".join(f"{s}" for s in sectors))

    # This will download and pre-cache the timeseries fits files. We don't open
    # them here as we may have to apply different settings per sector.
    fits_dir = Path.cwd() / "cache" / re.sub(r'[^\w\d-]', '_', target.lower())
    pipeline.find_lightcurves(search_term, fits_dir, sectors,
                              mission = target_cfg.get("mission", "TESS"),
                              author=target_cfg.get("author", "SPOC"),
                              exptime=target_cfg.get("exptime", None),
                              verbose=False)
    lcs = []
    for sector in sectors:
        # Now open & pre-process each LC directly.
        sector_cfg = sector_config_from_target(sector, target_cfg)
        lcs +=[_prepare_lc_for_target_sector(search_term, sector, sector_cfg, fits_dir, verbose)]
    if len(lcs) > 1:
        return (LightCurveCollection(lcs).stitch(), len(sectors))
    return (lcs[0], 1)


def _prepare_lc_for_target_sector(search_term: str,
                                 sector: int,
                                 config: Dict[str, any],
                                 fits_dir: Path,
                                 verbose: bool=True) -> LightCurve:
    """
    Will find and load the requested target/sector light curve then mask, bin
    and append the rectified delta_mag and delta_mag_err columns. It will read
    the information required for these steps from the passed sector config.

    :search_term: the name or id of the chosen target system
    :sector: which sector
    :target_cfg: the target's config dictionary, as read from the config json
    :fits_dir: the location of the local cache for the downloaded MAST fits assets
    :verbose: whether to write detailed progress information to stdout
    :returns: a LightCurve object with prepared light curve data
    """
    lc = pipeline.find_lightcurves(search_term,
                                   fits_dir,
                                   [sector],
                                   config.get("mission", "TESS"),
                                   config.get("author", "SPOC"),
                                   config.get("exptime", "short"),
                                   config.get("flux_column", "sap_flux"),
                                   config.get("quality_bitmask", "default"),
                                   verbose=verbose)[0]

    if verbose:
        print(f"Opened light curve fits for {lc.meta['OBJECT']} sector {lc.meta['SECTOR']}")

    # Here we mask out any "bad" data which may adversely affect detrending and subsequent processes
    lc = pipeline.apply_invalid_flux_masks(lc, verbose)
    mask_time_ranges = config.get("quality_masks", None) or []
    if mask_time_ranges:
        if verbose:
            print(f"Applying {len(mask_time_ranges)} quality time range mask(s)...", end="")
        lc = pipeline.apply_time_range_masks(lc, mask_time_ranges, verbose)

    time_bin_seconds = (config.get("bin_time", None) or 0) * u.s
    if time_bin_seconds > 0 * u.s:
        lc = pipeline.bin_lightcurve(lc, time_bin_seconds, verbose)

    # Optionally apply some smoothing/flatten. We need to mask out the eclipses.
    flatten_kwargs = config.get("flatten", None)
    if flatten_kwargs:
        if "mask_time_ranges" in flatten_kwargs:
            flatten_kwargs["period"] = config["period"] * u.d
        lc = pipeline.flatten_lightcurve(lc, verbose=verbose, **flatten_kwargs)

    pipeline.append_magnitude_columns(lc, "delta_mag", "delta_mag_err")

    # Detrending and rectifying to differential mags
    gap_th = config.get("detrend_gap_threshold", None)
    detrend_ranges = [*pipeline.find_lightcurve_segments(lc, gap_th or 10000, return_times=True)]
    if verbose:
        print("Will detrend (and rectify by subracting trends) over the following range" +
              (f"(s) (detected on gaps > {gap_th} d)" if gap_th else "") +
              f" [{lc.time.format}]: " +
              ", ".join(f"{r[0].value:.6f}-{r[1].value:.6f}" for r in detrend_ranges))
    for detrend_range in detrend_ranges:
        mask = (lc.time >= np.min(detrend_range)) & (lc.time <= np.max(detrend_range))
        lc["delta_mag"][mask] -= pipeline.fit_polynomial(lc.time[mask],
                                                         lc["delta_mag"][mask],
                                                         config.get("detrend_order", 2),
                                                         config.get("detrend_iterations", 2),
                                                         config.get("detrend_sigma_clip", 1.0),
                                                         verbose=verbose)

    # Rather than being "bad" these are just data we no longer need, however we trim these
    # after detrending so the polynomial has plenty of "good" data to fit to.
    mask_time_ranges = config.get("trim_masks", None) or []
    if mask_time_ranges:
        if verbose:
            print(f"Applying {len(mask_time_ranges)} trim time range mask(s)...", end="")
        lc = pipeline.apply_time_range_masks(lc, mask_time_ranges, verbose)

    return lc
