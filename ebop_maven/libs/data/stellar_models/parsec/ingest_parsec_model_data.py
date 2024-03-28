""" Module to ingest PARSEC DAT evo tracks, interpolate and write to default data file """
from inspect import getsourcefile
from pathlib import Path
import sys

import astropy.units as u

# Hack so this script imports code from parent folder structure.
# pylint: disable=import-error, wrong-import-position
this_dir = Path(getsourcefile(lambda:0)).parent
sys.path.append(f"{(this_dir / '../../..').resolve()}")
from stellarmodels import ParsecStellarModels

# Using the PARSEC 1.2s evolutionary tracks available from following the
# "Evolutionay tracks" link on https://people.sissa.it/~sbressan/parsec.html
# Specifically, this builds a data file for solar metallicity based on
# https://people.sissa.it/~sbressan/CAF09_V1.2S_M36_LT/Z0.014Y0.273.tar.gz
# This glob ignores HB tracks and a few with very slight variation in initial mass.
dat_files = this_dir.glob("**/Z0.014*_M???.??0.DAT")
Z, Y = 0.014, 0.273

print(f"About to overwrite to the default models file:\t'{ParsecStellarModels.default_data_file}'")
if input("Are you sure (yes or [NO])? ").lower() in ["y", "yes"]:

    # This does the heavy lifting of parsing the source data for what's needed
    # and then writing (with overwrite) to the default data file.
    out_data_file = ParsecStellarModels.create_model_data_file(dat_files)

    # Now test the newly created model file with some well known lookups
    models = ParsecStellarModels()

    print("\nSummary")
    print(f"A total of {models.row_count} data row(s).")
    print(f"Z values: {models.list_distinct_z_values()}")
    print(f"Y values: {models.list_distinct_y_values()}")
    print(f"Initial masses: {models.list_distinct_initial_masses()}")

    # log10(4.6 Gyr)~9.7 and log10(1.7 Gyr)~9.2
    for (init_mass, age) in [
        (1.0 * u.solMass, 9.7 * u.dex(u.yr)),
        (1.55 * u.solMass, 9.2 * u.dex(u.yr)),
        (1.3 * u.solMass, 9.2 * u.dex(u.yr))]:

        mass, radius, t_eff, lum, logg = models.lookup_stellar_parameters(Z, Y, init_mass, age)
        print(f"lookup_stellar_parameters({Z}, {Y}, {init_mass}, {age}):",
              f"M={mass:.3f}, R={radius:.3f}, TE={t_eff:.3f}, L={lum:.3f}, log(g)={logg:.3f}")

else:
    print("Ingest aborted.")
