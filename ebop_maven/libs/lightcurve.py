""" Utility functions for Light-curves. """
from typing import Union, List, Iterable
from pathlib import Path
import math

import numpy as np
from astropy.io import fits

import lightkurve as lk
from lightkurve.collections import LightCurveCollection

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
    # pylint: disable=too-many-variables, unnecessary-lambda-assignment
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
