""" Module publishing methods for lookup up Limb Darkening coefficients. """
from inspect import getsourcefile
from pathlib import Path
from typing import Tuple, Iterable
from functools import lru_cache

import pandas as pd
from pandas import DataFrame
import astropy.units as u
from astropy.units import Quantity, quantity_input

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
    try:
        # Traverse the indices finding the nearest match
        # Unfortunately, get_indexer([(a, b)], method="nearest") not supported on multiindex
        index_values = __find_nearest_index_values(__tess_power2_coeffs_table(),
                                                   (logg.value, t_eff.to(u.K).value))

        row = __tess_quad_ld_coeffs_table().loc[index_values]
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
    try:
        # Traverse the indices finding the nearest match
        # Unfortunately, get_indexer([(a, b)], method="nearest") not supported on multiindex
        index_values = __find_nearest_index_values(__tess_power2_coeffs_table(),
                                                   (logg.value, t_eff.to(u.K).value))

        row = __tess_power2_coeffs_table().loc[index_values]
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

def __find_nearest_index_values(df: DataFrame, suggested_values: Iterable[float]) -> Tuple[float]:
    """
    Takes the suggested index values and finds the nearest matches within the dataframe's
    multiindex returning them as a tuple which can be used directly in a call to loc[]
    """
    loc_values = list(suggested_values)
    for value_ix, suggested_value in enumerate(suggested_values):
        # pylint: disable=cell-var-from-loop
        if value_ix == 0:
            index_values = df.index.levels[0].values
        elif value_ix == 1:
            # We get a warning if loc is a single item tuple of form (value, ) from tuple(value)
            index_values = df.loc[(loc_values[0])].index.values
        else:
            index_values = df.loc[tuple(loc_values[:value_ix])].index.values
        loc_values[value_ix] = min(index_values, key=lambda v: abs(v - suggested_value))
    return tuple(loc_values)
