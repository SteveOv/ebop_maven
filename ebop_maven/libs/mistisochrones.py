""" A simple class for querying MIST Isochrones """
from typing import List
from pathlib import Path
from inspect import getsourcefile
import re

import numpy as np

from .data.stellar_models.mist.read_mist_models import ISO

class MistIsochrones():
    """
    This class wraps one or more MIST isochrone files and exposes functions to query them.
    """
    _this_dir = Path(getsourcefile(lambda:0)).parent

    def __init__(self, metallicities: list[float]=None) -> None:
        """
        Initializes a MistIsochrones class.

        :metallicities: optional list of metallicities to load by feh value, or any found if not set
        """
        isos_dir = self._this_dir / "data/stellar_models/mist/MIST_v1.2_vvcrit0.4_basic_isos"
        iso_files = sorted(isos_dir.glob("*.iso"))

        # Index the iso files on their [Fe/H]
        self._isos: dict[float, List] = {}
        feh_file_pattern = re.compile(r"feh_(?P<pm>m|p)(?P<feh>[0-9\.]*)_", re.IGNORECASE)
        for iso_file in iso_files:
            match = feh_file_pattern.search(iso_file.stem)
            if match and "feh" in match.groupdict():
                pm = -1 if (match.group("pm") or "p") == "m" else 1
                feh = float(match.group("feh")) * pm
                if metallicities is None or feh in metallicities:
                    self._isos[feh] = ISO(f"{iso_file.resolve()}")
            else:
                raise Warning(f"Unexpected iso file name format: {iso_file}")


    def list_metallicities(self) -> np.ndarray:
        """
        List the distinct isochrone metallicity values available.
        """
        return np.array(list(self._isos.keys()))


    def list_ages(self, feh: float, min_phase: float=0.0, max_phase: float=9.0) -> np.ndarray:
        """
        List the distinct log10(age yr)s within the isochrones matching the passed metallicity.
        Only ages with records for stars within the min and max phases are returned.

        The supported phase values are (from the MIST documentation):
        -1=PMS, 0=MS, 2=RGB, 3=CHeB, 4=EAGB, 5=TPAGB, 6=postAGB, 9=WR

        :feh: the metallicity key value - used to select the appropiate isochrone
        :min_phase: only ages with at least one star at this or a later phase are returned
        :max_phase: only ages with at least one star at this or an earlier phase are returned
        :returns: the list of ages with at least some stars within the min and max phases
        """
        # We only want age blocks which contain stars within our chosen phase criteria
        iso = self._isos[feh]
        ages = [ab["log10_isochrone_age_yr"][0] for ab in iso.isos if ab["phase"][-1] >= min_phase
                                                                    or ab["phase"][0] <= max_phase]
        return np.array(ages)


    def list_initial_masses(self, feh: float, age: float,
                            min_phase: float=0.0, max_phase: float=9.0,
                            min_mass: float=None, max_mass: float=None) -> np.ndarray:
        """
        Lists the distinct initial masses which are available for the chose set of
        age, phase and mass criteria.

        The supported phase values are (from the MIST documentation):
        -1=PMS, 0=MS, 2=RGB, 3=CHeB, 4=EAGB, 5=TPAGB, 6=postAGB, 9=WR

        :feh: the metallicity key value - used to select the appropiate isochrone
        :min_phase: only masses where the star is in at least this phase are returned
        :max_phase: only masses where the star is in at mose this phase are returned
        :min_mass: only masses above this value are returned
        :max_phase: only masses below this value are returned
        """
        # pylint: disable=too-many-arguments
        iso = self._isos[feh]
        age_block = iso.isos[iso.age_index(age)]
        if min_phase:
            age_block = age_block[age_block["phase"] >= min_phase]
        if max_phase:
            age_block = age_block[age_block["phase"] <= max_phase]
        if min_mass:
            age_block = age_block[age_block["initial_mass"] >= min_mass]
        if max_mass:
            age_block = age_block[age_block["initial_mass"] <= max_mass]
        return np.array(age_block["initial_mass"])


    def lookup_stellar_params(self, feh: float, age: float, initial_mass: float,
                              cols: list[str|int]) -> dict[str, float]:
        """
        Will retrieve the values of the requested columns from the isochrone keyed on the
        metallicity with the requested initial mass and age.

        :feh: the metallicity key value - used to select the appropiate isochrone
        :age: the log age of the star
        :initial_mass: the initial mass of the star
        :cols: the columns to return
        :returns: a dictionary
        """
        # pylint: disable=too-many-arguments
        iso = self._isos[feh]
        age_block = iso.isos[iso.age_index(age)]
        row = age_block[age_block["initial_mass"] == initial_mass]
        return { col: row[col][0] for col in cols }

    def lookup_zams_params(self, feh: float, cols: list[str|int]) -> np.ndarray:
        """
        Will get the values for the chosen metallicity and columns across the
        zero age main-sequence (ZAMS) equivalent evolutionary point (EEP).

        :feh: the metallicity key value - used to select the appropiate isochrone
        :cols: the columns to return
        :returns: a 2-d NDArray[cols, rows] with cols in the input order
        """
        # We look for the first M-S EEP (equivalent evolutionary point) which is 202.
        return self._get_eep_column_values(feh, 202, 0.0, cols)

    def lookup_tams_params(self, feh: float, cols: list[str|int]) -> np.ndarray:
        """
        Will get the values for the chosen metallicity and columns across the
        terminal age main-sequence (TAMS) equivalent evolutionary point (EEP).

        :feh: the metallicity key value - used to select the appropiate isochrone
        :cols: the columns to return
        :returns: a 2-d NDArray[cols, rows] with cols in the input order
        """
        # We look for the last M-S EEP (equivalent evolutionary point) which is 453
        return self._get_eep_column_values(feh, 453, 0.0, cols)

    def _get_eep_column_values(self, feh: float, eep: int, phase: float, cols: list[str|int]):
        # Two stage process as imposing both eep and phase criteria may yield empty rows
        iso = self._isos[feh]
        rows = (ab[(ab["EEP"]==eep) & (ab["phase"]==phase)] for ab in iso.isos if eep in ab["EEP"])
        return np.array([list(row[0][cols]) for row in rows if len(row) > 0]).transpose()
