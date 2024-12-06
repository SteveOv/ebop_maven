""" A module of lightcurve related helper functions for use throughout the tests. """
# pylint: disable=no-member
from inspect import getsourcefile
from pathlib import Path
from argparse import Namespace

import numpy as np
import astropy.units as u
from astropy.time import Time
import lightkurve as lk

# Use this as it will use previously downloaded fits if available, avoiding remote MAST calls.
from traininglib.pipeline import find_lightcurves

TEST_DATA_DIR = Path(getsourcefile(lambda:0)).parent / "../../../../cache/test_data"

""" A dictionary of known lightkurve downloadable targets. """
KNOWN_TARGETS = {
    "CW Eri": Namespace(**{ # Easy light-curve
        "tic": 98853987,
        "sector": 31,
        "period": 2.7283712677 * u.d,
        "epoch_time": Time(2152.1428924222, format="btjd", scale="tdb"),
        "ecosw": 0.00502,
        "esinw": -0.0121,
        "ecc": 0.0131,
        "expect_phase2": 0.5032,
        "expect_width2": 0.976,
    }),
    "RR Lyn": Namespace(**{ # Early secondary eclipses
        "tic": 11491822,
        "sector": 20,
        "period": 9.946591113 * u.d,
        "epoch_time": Time(1851.925277662, format="btjd", scale="tdb"),
        "expect_phase2": 0.45,
        "expect_width2": 1.2,
    }),
    "IT Cas": Namespace(**{ # Eccentric, late secondary, primary/secondary similar depths
        "tic": 26801525,
        "sector": 17,
        "period": 3.8966513 * u.d,
        "epoch_time": Time(1778.3091293396, format="btjd", scale="tdb"),
        "expect_phase2": 0.55,
        "expect_width": 1.0,      
    }),
    "AN Cam": Namespace(**{ # Very late secondary eclipses, near phase 0.8
        "tic": 103098373,
        "sector": 25,
        "author": "TESS-SPOC",
        "exptime": "long",
        "period": 20.99842 * u.d,
        "epoch_time": Time(1992.007512423, format="btjd", scale="tdb"),
        "expect_phase2": 0.78,
        "expect_width2": 1.1,
    }),
    "V889 Aql": Namespace(**{ # Lower (600 s) cadence, highly eccentric and small mid-sector gap
        "tic": 300000680,
        "sector": 40,
        "author": "TESS-SPOC",
        "exptime": 600,
        "period": 11.120757 * u.d,
        "epoch_time": Time(2416.259790, format="btjd", scale="tdb"),
        "expect_phase2": 0.35,
        "expect_width2": 1.9,
    }),
}

def load_lightcurve(target: str, setup_mag_columns: bool=True) -> lk.LightCurve:
    """
    Loads a LightCurve for the requested target.
    """
    params = KNOWN_TARGETS[target]
    tic = f"TIC{params.tic}"
    lc = find_lightcurves(search_term=tic,
                          download_dir=TEST_DATA_DIR / tic,
                          sectors=params.sector,
                          mission=params.__dict__.get("mission", "TESS"),
                          author=params.__dict__.get("author", "SPOC"),
                          exptime=params.__dict__.get("exptime", "short"),
                          flux_column=params.__dict__.get("flux_column", "sap_flux"),
                          quality_bitmask="hardest")[0]

    if setup_mag_columns:
        _setup_mag_columns(lc, in_place=True)
    return lc


def _setup_mag_columns(lc: lk.LightCurve, in_place: bool=True):
    """
    Calculates and appends the delta_mag and delta_mag_err columns
    to a copy of the passed LightCurve with the copy being returned.
    """
    if not in_place:
        lc = lc.copy()
    fmax = lc.flux.max().value
    lc["delta_mag"] = u.Quantity(2.5 * np.log10(fmax/lc.flux.value) * u.mag)
    lc["delta_mag_err"] = u.Quantity(
        2.5
        * 0.5
        * np.abs(
            np.subtract(
                np.log10(np.add(lc.flux.value, lc.flux_err.value)),
                np.log10(np.subtract(lc.flux.value, lc.flux_err.value))
            )
        )
        * u.mag)
    if not in_place:
        return lc
