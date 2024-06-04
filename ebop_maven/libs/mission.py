""" Photometry missions. """
from inspect import getsourcefile
from pathlib import Path
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas import DataFrame

# This pylint disable makes it overlook its issue with astropy const aliases
# pylint: disable=no-member
import astropy.constants as const
import astropy.units as u
from astropy.units import quantity_input

class Mission(ABC):
    """ Base class for photemetry mission characteristics. """
    COL_NAMES = ["lambda", "coefficient"]

    _this_dir = Path(getsourcefile(lambda:0)).parent

    @classmethod
    @lru_cache
    def get_instance(cls, mission_name: str, **kwargs):
        """
        A factory method for getting an instance of a chosen Mission subclass.
        Selects the Mission with a name containing the passed text (i.e.
        TES matches Tess). Raises a KeyError if no match made.

        :model_name: the name of the Mission
        :kwargs: the arguments for the Mission's initializer
        :returns: a cached instance of the chosen Mission
        """
        for subclass in cls.__subclasses__():
            if mission_name.strip().lower() in subclass.__name__.lower():
                return subclass(**kwargs)
        raise KeyError(f"No Mission subclass named like {mission_name}")

    @classmethod
    @abstractmethod
    def get_response_function(cls) -> DataFrame:
        """
        Get the mission's response function, wavelength against efficiency

        :returns: pandas DataFrame indexed on lambda with a coefficient column
        """

    @classmethod
    def get_default_bandpass(cls) -> (u.nm, u.nm): # type: ignore
        """
        Gets the default bandpass we use for this mission in nm
        
        :returns: the bandpass in the form (from Quantity nm, to Quantity nm)
        """
        response = cls.get_response_function()
        return (min(response.index) * u.nm, max(response.index) * u.nm)

    @classmethod
    @u.quantity_input
    def expected_brightness_ratio(cls,
                                  t_eff_a: u.K,
                                  t_eff_b: u.K,
                                  bandpass: (u.nm, u.nm) = None # type: ignore
                                  ) -> float:
        """
        Calculate the brightness ratio of two stars with the passed effective
        temperatures over the requested bandpass making use of this mission's
        response function.

        :t_eff_a: effective temperature as a Quantity of K of the primary star
        :t_eff_b: effective temperature as a Quantity of K of the secondary star
        :bandpass: the range of wavelengths as (u.nm, u.nm) to calculate over
        or, if None, the result of get_default_bandpass() will be used
        :returns: simple ratio of the secondary/primary brightness
        """
        t_eff_a = t_eff_a.to(u.K)
        t_eff_b = t_eff_b.to(u.K)
        if bandpass is None:
            bandpass = cls.get_default_bandpass()

        response = cls.get_response_function()
        mask = (response.index >= min(bandpass).to(u.nm).value) \
                & (response.index <= max(bandpass).to(u.nm).value)

        radiance1, radiance2 = 0., 0.
        for lam in response[mask].index:
            coeff = response.loc[lam].coefficient
            lam *= u.nm
            radiance1 += coeff * cls.__bb_spectral_radiance(t_eff_a, lam).value
            radiance2 += coeff * cls.__bb_spectral_radiance(t_eff_b, lam).value
        return radiance2 / radiance1

    @classmethod
    @quantity_input
    def __bb_spectral_radiance(cls, temperature: u.K, wavelegth: u.nm) \
                                            -> u.W / u.m**2 / u.sr / u.nm: # type: ignore
        """
        Calculates the blackbody spectral radiance:
        power / (area*solid angle*wavelength) at a given temperature and wavelength.

        Uses: B_λ(T) = (2hc^2)/λ^5 * 1/(exp(hc/λkT)-1)
        
        :temperature: the temperature of the body in K
        :wavelength: the wavelength of the radiation.
        :returns: the calculated radiance in units of W / m^2 / sr / nm
        """
        c = const.c
        h = const.h
        k_B = const.k_B # pylint: disable=invalid-name
        pt1 = np.divide(np.multiply(2, np.multiply(h, np.power(c, 2))),
                        np.power(wavelegth, 5))
        inr = np.divide(np.multiply(h, c),
                        np.multiply(wavelegth, np.multiply(k_B, temperature)))
        pt2 = np.reciprocal(np.subtract(np.exp(inr), 1))
        return np.multiply(pt1, pt2) / u.sr


class Tess(Mission):
    """ Characteristics of the TESS mission. """
    def __init__(self):
        pass

    @classmethod
    def get_default_bandpass(cls) -> (u.nm, u.nm): # type: ignore
        return (600 * u.nm, 1000 * u.nm)

    @classmethod
    @lru_cache
    def get_response_function(cls) -> DataFrame:
        """
        The TESS response function, wavelength [nm] against efficiency coeff

        :returns: DataFrame indexed on lambda [nm] with a coefficient column
        """
        file = cls._this_dir \
                        / "data/missions/tess/tess-response-function-v2.0.csv"
        return pd.read_csv(file,
                           comment="#",
                           delimiter=",",
                           names=Mission.COL_NAMES,
                           index_col=0)


class Kepler(Mission):
    """ Characteristics of the Kepler mission. """
    def __init__(self):
        pass

    @classmethod
    def get_default_bandpass(cls) -> (u.nm, u.nm): # type: ignore
        return (420 * u.nm, 900 * u.nm)

    @classmethod
    @lru_cache
    def get_response_function(cls) -> DataFrame:
        """
        The Kepler response function, wavelength [nm] against efficiency coeff

        :returns: DataFrame indexed on lambda [nm] with a coefficient column
        """
        file = cls._this_dir \
                        / "data/missions/kepler/kepler_response_hires1.txt"
        return pd.read_csv(file,
                           comment="#",
                           delim_whitespace=True,
                           names=Mission.COL_NAMES,
                           index_col=0)
