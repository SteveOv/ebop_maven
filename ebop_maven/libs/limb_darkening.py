""" Module publishing methods for lookup up Limb Darkening coefficients. """
from inspect import getsourcefile
from pathlib import Path
from typing import Tuple
from functools import lru_cache
import numpy as np
import pandas as pd
from pandas import DataFrame
import astropy.units as u
from astropy.units import Quantity, quantity_input
from numba import jit

_this_dir = Path(getsourcefile(lambda:0)).parent

@quantity_input(logg=[u.dex, u.dex(u.cm / u.s**2)], t_eff=u.K)
def lookup_tess_quad_ld_coeffs(logg: Quantity, t_eff: Quantity) \
                                -> Tuple[float, float]:
    """
    Get the Claret (A/A&A/618/A20) TESS quad limb darkening coefficients
    nearest to the passed log(g) and T_eff combination. Will raise a
    KeyError if no close match found.

    :logg: the requested logg in dex - nearest value will be used
    :t_eff: the requested T_eff in K - nearest value will be used
    :returns: tuple (a, b) where a is the linear and b the quadratic coefficient
    or will raise a KeyError if no match made
    """
    # Find the nearest match to requested log(g) and T_eff values
    logg = __round_to_nearest(logg.value, 0.5)
    t_eff = t_eff.to(u.K).value
    if t_eff < 2300:
        t_eff = 2300
    elif 4900 < t_eff < 5100:
        # Special case; the data is missing values for t_eff == 5000 K
        # We need to force the requested value to 4900 or 5100 K as appropriate
        t_eff = 4900. if t_eff < 5000 else 5100.
    elif t_eff < 7000:
        t_eff = __round_to_nearest(t_eff, 100)
    elif t_eff < 12000:
        t_eff = __round_to_nearest(t_eff, 200)
    else:
        t_eff = 12000

    try:
        row = __tess_quad_ld_coeffs_table().loc[(logg, t_eff)]
    except KeyError as exc:
        raise KeyError(f"No quad coeffs for logg={logg} and t_eff={t_eff} K") from exc

    return row["a"], row["b"]

@quantity_input(logg=[u.dex, u.dex(u.cm / u.s**2)], t_eff=u.K)
def lookup_tess_pow2_ld_coeffs(logg: Quantity, t_eff: Quantity) \
                                -> Tuple[float, float]:
    """
    Get the Claret+ (A/A&A/674/A63) TESS power-2 limb darkening coefficients
    nearest to the passed log(g) and T_eff combination. Will raise a
    KeyError if no close match found.

    :logg: the requested logg in dex - nearest value will be used
    :t_eff: the requested T_eff in K - nearest value will be used
    :returns: tuple with the (g, h) power-2 LD coefficients
    or will raise a KeyError if no match made
    """
    # Find the nearest match to requested log(g) and T_eff values
    logg = __round_to_nearest(logg.value, 0.5)
    t_eff = t_eff.to(u.K).value
    if t_eff < 2300:
        t_eff = 2300
    elif 4900 < t_eff < 5100:
        # Special case; the data is missing values for t_eff == 5000 K
        # We need to force the requested value to 4900 or 5100 K as appropriate
        t_eff = 4900. if t_eff < 5000 else 5100.
    elif t_eff < 7000:
        t_eff = __round_to_nearest(t_eff, 100)
    elif t_eff < 12000:
        t_eff = __round_to_nearest(t_eff, 200)
    else:
        t_eff = 12000

    try:
        row = __tess_power2_coeffs_table().loc[(logg, t_eff)]
    except KeyError as exc:
        raise KeyError(f"No pow2 coeffs for logg={logg} and t_eff={t_eff} K") from exc
    return row["g"], row["h"]

# Using the lru_cache as a way of achieving lazy loading of these dataframes
@lru_cache
def __tess_quad_ld_coeffs_table() -> DataFrame:
    """ Publishes the DataFrame containing the TESS quad LD coefficients. """
    # These are the PHOENIX-COND models, solar metallicity, v.tu=2 km/s for TESS
    data_file = _this_dir / "data/limb_darkening/quad/J_A+A_618_A20/table5.dat"
    return pd.read_csv(data_file,
                       comment="#",
                       sep=r"\s+",
                       usecols=[0, 1, 2, 4, 5, 9],
                       names=["logg", "Teff", "Z", "a", "b", "Sys"],
                       index_col=["logg", "Teff"])

@lru_cache
def __tess_power2_coeffs_table() -> DataFrame:
    """ Publishes the DataFrame containing the TESS power-2 LD coefficients. """
    # These are the M1 coeffs derived from PHOENIX spherically symmetric models
    # for Gaia, Kepler, TESS & CHEOPS. TESS g & h coeffs on cols 8 & 14
    data_file = _this_dir / "data/limb_darkening/pow-2/J_A+A_674_A63/table1.dat"
    return pd.read_csv(data_file,
                    comment="#",
                    sep=r"\s+",
                    usecols=[0, 1, 2, 8, 14],
                    names=["logg", "Teff", "Z", "g", "h"],
                    index_col=["logg", "Teff"])

@jit(nopython=True)
def __round_to_nearest(value, nearest=1.):
    """
    Will round the passed value to the nearest value to the passed
    nearest argument. 
    For example round_to_nearest(23.74, 0.5) will return 23.5
    wheras round_to_nearewst(23.75, 0.5) will return 24.

    :value: the value to round
    :nearest: the target to round "to the nearest". Defaults to 1.
    """
    if nearest == 1.:
        result = np.round(value)
    else:
        mod = np.mod(value, nearest)
        result = np.subtract(value, mod)
        if np.add(mod, mod) >= nearest:
            result = np.add(result, nearest)
    return result
