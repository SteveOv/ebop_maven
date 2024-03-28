""" Provides lookup access to stellar models' characteristics """
from typing import List, Tuple, Dict
from pathlib import Path
from inspect import getsourcefile
from abc import ABC, abstractmethod
from functools import lru_cache
import re

import numpy as np
from scipy.interpolate.interpolate import interp1d
import pandas as pd
import astropy.units as u
import astropy.constants as const
from astropy.units import quantity_input, Quantity

# From MIST resources under MIT Licence
# https://github.com/jieunchoi/MIST_codes/blob/master/scripts/
from .data.stellar_models.mist.read_mist_models import EEP

# The non-member pylint "disable" is to address its issue with astropy.constants
# pylint: disable=no-member
# pyling: disable=too-many-arguments

_this_dir = Path(getsourcefile(lambda:0)).parent

class StellarModels(ABC):
    """ Base class for a Stellar Model """
    _COL_Z = "Z"
    _COL_Y = "Y"
    _COL_INIT_MASS = "INIT_MASS"
    _COL_LOG_AGE = "LOG_AGE"
    _COL_MASS = "M"
    _COL_RADIUS = "R"
    _COL_TE = "TE"
    _COL_LUM = "L"
    _COL_LOGG = "LOGG"
    _COL_PHASE = "PHASE"
    _INDEX_COLS = [_COL_Z, _COL_Y, _COL_INIT_MASS, _COL_LOG_AGE]
    _units = {
        _COL_Z: 1,
        _COL_Y: 1,
        _COL_INIT_MASS: u.solMass,
        _COL_LOG_AGE: u.dex(u.yr),
        _COL_MASS: u.solMass,
        _COL_RADIUS: u.solRad,
        _COL_TE: u.K,
        _COL_LUM: u.solLum,
        _COL_LOGG: u.dex(u.cm / u.s**2),
        _COL_PHASE: 1
    }
    _models_df = None

    def __init__(self, data_file: Path) -> None:
        """
        Initializes a new StellarModels instance.
        Raises a FileNotFoundError if the indicated data_file doesn't exist.

        :data_file: Path the the data file constaining the models' data.
        """
        print(f"Initializing {self.__class__.__name__} on model data in '{data_file}'")
        self._models_df = pd.read_pickle(data_file, compression="infer")
        self._models_df.set_index(self._INDEX_COLS, inplace=True)

    @classmethod
    @lru_cache
    def get_instance(cls, instance_name: str, **kwargs):
        """
        A factory method for getting an instance of a chosen StellarModels
        subclass. Select the model with a name containing the passed text (i.e.
        Parsec matches ParsecStellarModels). Raises a KeyError if no match made.

        :instance_name: the name of the subclass
        :kwargs: the arguments for the model's initializer
        :returns: a cached instance of the chosen StellarModel
        """
        for subclass in cls.__subclasses__():
            if instance_name.strip().lower() in subclass.__name__.lower():
                return subclass(**kwargs)
        raise KeyError(f"No StellarModels subclass named like {instance_name}")

    @classmethod
    def create_model_data_file(cls,
                               source_files: List[Path],
                               out_data_file: Path = None) -> Path:
        """
        Creates a data file suitable for reading by this class for lookups
        of basic stellar parameters.

        :source_files: this files to read from
        :out_data_file: the data file to create, or None to use the default file
        :returns: the path of out_data_file
        """
        source_files = sorted(source_files)     # Also unpacks if a generator (ie: from a glob)
        source_file_count = len(source_files)
        dfs = []

        for ix, source_file in enumerate(source_files, start=1):
            print(f"Processing file {ix} of {source_file_count}: {source_file}")
            df = cls._dataframe_from_model_file(source_file) # To be implemented by each subclass
            print(f"Ingested {len(df)} model row(s) from {source_file.name}\n")
            dfs.append(df)

        out_df = pd.concat(dfs, ignore_index=True)
        out_df.sort_values(by=cls._INDEX_COLS, inplace=True)

        if out_data_file is None:
            out_data_file = cls.default_data_file
        print(f"Writing {len(out_df)} row(s) of output data to pickle file: {out_data_file}")
        out_df.to_pickle(out_data_file, compression="infer")
        return out_data_file

    @classmethod
    @abstractmethod
    def _dataframe_from_model_file(cls, source_file: Path) -> pd.DataFrame:
        """
        Will parse selected contents of the passed source file into a DataFrame
        for use as a lookup of models data.

        :source_file: the selected source file in the expected format
        :returns: a DataFrame with this class's expected columns populated
        """

    @property
    def row_count(self) -> int:
        """ The number of model rows available for lookup. """
        return len(self._models_df) if self._models_df is not None else None

    @quantity_input(initial_mass=u.solMass, age=u.dex(u.yr))
    def lookup_stellar_parameters(self,
                                  z: float,
                                  y: float,
                                  initial_mass: Quantity,
                                  age: Quantity) \
                                    -> Tuple[Quantity, Quantity, Quantity, Quantity]:
        """
        Get the current mass, radius, effective temperature, luminosity and
        log(g) from the evolutionary models for a star with the requested Z, Y,
        initial mass and current age.

        Raises a KeyError if no match found for the criteria

        :initial_mass: the initial mass of the star to lookup
        :age: the current age of the star to lookup
        :z: the Z fraction (metals) of the star to lookup [0.014]
        :y: the Y fraction (He) of the star to lookup [0.273]
        :returns: the (mass, radius, T_eff, lum, logg) in (solMass, solRad, K, solLum, dex)
        """
        # Currently, the only param we don't return is PHASE, so the quickest
        # approach (bypasses lots of checks) is to get them all and ignore PHASE
        return tuple(self.lookup_parameters(z, y, initial_mass, age)[:-1])

    @quantity_input(initial_mass=u.solMass, age=u.dex(u.yr))
    def lookup_parameters(self,
                          z: float,
                          y: float,
                          initial_mass: Quantity,
                          age: Quantity,
                          parameter_names: List[str] = None) -> List[any]:
        """
        Get the values & units of the requested parameters for a star with the
        requested Z, Y, initial mass and current age values. If nothing found
        a KeyError is raised. If a field name is unknown a ValueError is raised.

        :initial_mass: the initial mass of the star to lookup
        :age: the current age of the star to lookup
        :z: the Z fraction (metals) of the star to lookup [0.014]
        :y: the Y fraction (He) of the star to lookup [0.273]
        :parameter_names: what to lookup in the order they are required. If
        None then all lookup params will be returned in the order they are held.
        :returns: a list of the requested values & units in the order requested.
        """
        initial_mass = np.round(initial_mass.to(u.solMass).value, 3)
        age = age.to(u.dex(u.yr)).value
        if z is None:
            raise TypeError("z cannot be None")
        if y is None:
            raise TypeError("y cannot be None")
        try:
            # Will throw an KeyError if no matching index/row
            row = self._models_df.loc[(z, y, initial_mass, age)]
        except KeyError as exc:
            raise KeyError("No match found for lookup_stellar_parameters" + \
                           f"({z}, {y}, {initial_mass}, {age})") from exc

        if parameter_names is None:
            unames = row.index
        else:
            # Get the given names into upper case and check they are known
            # Anything not a string will cause a TypeError to be raised
            unames = list(map(str.upper, parameter_names))
            unknown = np.setdiff1d(unames, [*self._units.keys()])
            if len(unknown) > 0:
                raise ValueError(f"The parameter names {unknown} are unknown.")

        # Return the value in the order given and with appropriate units applied
        return [row[name] * self._units[name] for name in unames]

    def list_distinct_z_values(self, y_filter: np.double = None) -> List[np.double]:
        """
        Get a list of the distinct Z values found in the models data.

        :y_filter: optional Y value to filter results by
        :returns: list of matching Z values
        """
        filter_by = {self._COL_Y: y_filter} if y_filter else None
        return self._list_distinct_index_level_values("Z", filter_by)

    def list_distinct_y_values(self, z_filter: np.double = None) -> List[np.double]:
        """
        Get a list of the distinct Y values found in the models data.

        :z_filter: optional Z value to filter results by
        :returns: list of matching Y values
        """
        filter_by = {"Z": z_filter} if z_filter else None
        return self._list_distinct_index_level_values("Y", filter_by)

    def list_distinct_initial_masses(self) -> u.solMass:
        """
        Get a list of the distinct initial mass (M) values in the models data.

        :returns: list of matching mass values as a Quantity with data mass units
        """
        return self._list_distinct_index_level_values(self._COL_INIT_MASS) * u.solMass

    @quantity_input(initial_mass_filter=u.solMass)
    def list_distinct_ages(self, initial_mass_filter: Quantity = None) -> u.dex(u.yr):# type: ignore
        """
        Get a list of the distinct AGE values found in models data.

        :initial_mass_filter: optional initial mass value to filter results by
        :returns: list of matching age values as a Quantity with the data's age unit
        """
        if initial_mass_filter is not None:
            filter_by = { self._COL_INIT_MASS: initial_mass_filter.to(u.solMass).value }
        else:
            filter_by = None
        return self._list_distinct_index_level_values(self._COL_LOG_AGE, filter_by) * u.dex(u.yr)

    def _list_distinct_index_level_values(self,
                                          index_col_name: str,
                                          filter_by: Dict = None) \
                                                -> List[any]:
        """
        Helper function which lists the unique values of a named index field.

        :name: the name of the index field to enumerate
        :filter_by: dictionary of simple index_field_name: (==) value filters
        """
        if not filter_by:
            values = self._models_df.index.get_level_values(index_col_name)
        else:
            index_df = self._models_df.index.to_frame()
            mask = [True] * len(index_df)
            for fname, fval in filter_by.items():
                if fname and fval is not None:
                    fname = fname.upper()
                    mask &= index_df[fname] == fval

            values = index_df[mask][index_col_name]
        return list(values.unique())


class MistStellarModels(StellarModels):
    """ A class to lookup for stellar parameters derived from MIST evolutionary tracks """
    default_data_file = _this_dir / "data/stellar_models/mist/default.pkl.xz"

    def __init__(self, data_file: Path = None) -> None:
        """
        Initializes a new MistStellarModels instance.
        Raises a FileNotFoundError if the indicated data_file doesn't exist.

        :data_file: Path the the data file constaining the MIST model data.
        If None this will default to default_data_file
        """
        super().__init__(data_file if data_file is not None else self.default_data_file)

    @classmethod
    def _dataframe_from_model_file(cls, source_file: Path) -> pd.DataFrame:
        eep = EEP(f"{source_file}")
        z = round(eep.abun["Zinit"], 4)
        y = round(eep.abun["Yinit"], 4)
        initial_mass = round(eep.minit, 3)
        ages = eep.eeps["star_age"]

        print(f"\tParsing the MIST evo track for Z={z}, Y={y} & initial_mass={initial_mass} M_Sun",
              f"covering the age range {ages.min():.3e} to {ages.max():.3e} yr",
              f"and phase range {eep.eeps['phase'].min()} to {eep.eeps['phase'].max()}")

        # We'll interpolate the data at set ages and at the same time move the
        # ages on to a log scale and the R, TE and L off of a log scale.
        min_log_age = np.log10(ages.min())
        min_log_age = np.max([5.0, np.ceil(min_log_age*10)/10])
        max_log_age = np.log10(ages.max())
        max_log_age = np.min([12.0, np.floor(max_log_age*10)/10])

        # Force rounding to 1 d.p. on log(AGE) otherwise we may not get matches on lookup
        interp_log_ages = np.round(np.arange(min_log_age, max_log_age+0.1, 0.1), 1)
        interp_ages = np.power(10, interp_log_ages)

        print(f"\tInterpolating over log10(AGE [yr]) range from {min_log_age} to {max_log_age}")
        df = pd.DataFrame({
            # index columns
            cls._COL_Z:         np.full_like(interp_ages, z),
            cls._COL_Y:         np.full_like(interp_ages, y),
            cls._COL_INIT_MASS: np.full_like(interp_ages, initial_mass),
            cls._COL_LOG_AGE:   interp_log_ages,
            # lookup data
            cls._COL_MASS:      interp1d(ages, eep.eeps["star_mass"])(interp_ages),
            cls._COL_RADIUS:    interp1d(ages, np.power(10, eep.eeps["log_R"]))(interp_ages),
            cls._COL_TE:        interp1d(ages, np.power(10, eep.eeps["log_Teff"]))(interp_ages),
            cls._COL_LUM:       interp1d(ages, np.power(10, eep.eeps["log_L"]))(interp_ages),
        })

        # (Re)calculate log(g) for the newly interpolated radii & masses
        g = np.divide(np.multiply(const.G, df[cls._COL_MASS].values * u.solMass),
                      np.power(df[cls._COL_RADIUS].values * u.solRad, 2)).cgs
        df[cls._COL_LOGG] = np.log10(g.value)

        # For MIST data the phase field is effectively an integer encoded as a float and most
        # interped phases will land between (and will equal) two equal int phase values. The
        # edge case is where it falls between a phase transition leading to a fractional phase.
        # We can potentially use this as indicating a transition where params are in doubt,
        df[cls._COL_PHASE] = interp1d(ages, eep.eeps["phase"])(interp_ages)
        return df


class ParsecStellarModels(StellarModels):
    """ A class to lookup for stellar parameters derived from PARSEC evolutionary tracks """
    default_data_file = _this_dir / "data/stellar_models/parsec/default.pkl.xz"

    _R_sun_in_cm = const.R_sun.to(u.cm).value   # pylint: disable=invalid-name

    # Used to extract Z, Y, M (initial mass) & IS_HB values from DAT file name
    _file_name_match = re.compile(r"Z(?P<Z>[0-9]+[\.]?[0-9]*)"\
                                  r"Y(?P<Y>[0-9]+[\.]?[0-9]*)"\
                                  r"(?:.*)"\
                                  r"_M(?P<M>[0-9]+[\.]?[0-9]*)"\
                                  r"(?P<HB>(\.HB){0,1})", re.IGNORECASE)

    def __init__(self, data_file: Path = None) -> None:
        """
        Initializes a new ParsecStellarModels instance.
        Raises a FileNotFoundError if the indicated data_file doesn't exist.

        :data_file: Path the the data file constaining the MIST model data.
        If None this will default to ./data/stellar_models/parsec/default.pkl.xz
        """
        super().__init__(data_file if data_file is not None else self.default_data_file)

    @classmethod
    def _dataframe_from_model_file(cls, source_file: Path) -> pd.DataFrame:
        dat = np.genfromtxt(f"{source_file}", names=True, unpack=True, dtype=np.double)
        match = cls._file_name_match.search(source_file.stem)
        z = round(np.double(match.group("Z")), 4)
        y = round(np.double(match.group("Y")), 4)
        initial_mass = round(np.double(match.group("M")), 3)
        ages = dat["AGE"]

        print(f"\tParsing PARSEC evo track for Z={z}, Y={y} & initial_mass={initial_mass} M_Sun",
              f"covering the age range {ages.min():.3e} to {ages.max():.3e} yr",
              f"and phase range {dat['PHASE'].min()} to {dat['PHASE'].max()}")

        # We'll interpolate the data at set ages and at the same time move the ages on to a
        # log scale and the radius, effective temp and luminosity on to the linear scale.
        min_log_age = np.log10(ages.min()) if ages.min() > 0 else 0.
        min_log_age = np.max([5.0, np.ceil(min_log_age*10)/10])
        max_log_age = np.log10(ages.max())
        max_log_age = np.min([12.0, np.floor(max_log_age*10)/10])

        # Force rounding to 1 d.p. on log(AGE) otherwise we may not get matches on lookup
        interp_log_ages = np.round(np.arange(min_log_age, max_log_age+0.1, 0.1), 1)
        interp_ages = np.power(10, interp_log_ages)

        print(f"\tInterpolating over log10(AGE [yr]) range from {min_log_age} to {max_log_age}")

        df = pd.DataFrame({
            # index columns
            cls._COL_Z:         np.full_like(interp_ages, z),
            cls._COL_Y:         np.full_like(interp_ages, y),
            cls._COL_INIT_MASS: np.full_like(interp_ages, initial_mass),
            cls._COL_LOG_AGE:   interp_log_ages,
            # lookup data. The source LOG_R appears to be in dex(cm)
            # (see Bressan, 2012, Table 3 on pp135) so rescale to R_sun.
            cls._COL_MASS:      interp1d(ages, dat["MASS"])(interp_ages),
            cls._COL_RADIUS:    interp1d(ages, np.divide(np.power(10, dat["LOG_R"]),
                                                         cls._R_sun_in_cm))(interp_ages),
            cls._COL_TE:        interp1d(ages, np.power(10, dat["LOG_TE"]))(interp_ages),
            cls._COL_LUM:       interp1d(ages, np.power(10, dat["LOG_L"]))(interp_ages),
        })

        # (Re)calculate log(g) for the newly interpolated radii & masses
        g = np.divide(np.multiply(const.G, df[cls._COL_MASS].values * u.solMass),
                      np.power(df[cls._COL_RADIUS].values * u.solRad, 2)).cgs
        df[cls._COL_LOGG] = np.log10(g.value)

        # Unlike for MIST, the PARSEC PHASE field is a continuum so interpolation
        # of the phase values works naturally.
        df[cls._COL_PHASE] = interp1d(ages, dat["PHASE"])(interp_ages)
        return df
