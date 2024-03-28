""" Module to ingest MIST eep tracks, interpolate and write to the default data file """
from inspect import getsourcefile
from pathlib import Path
import sys

import astropy.units as u

# Hack so this script imports code from parent folder structure.
# pylint: disable=import-error, wrong-import-position
this_dir = Path(getsourcefile(lambda:0)).parent
sys.path.append(f"{(this_dir / '../../..').resolve()}")
from stellarmodels import MistStellarModels

# Using the basic stellar metallicity [Fe/H]==0 & v/crit==0.4 set of evolutionary tracks / EEPs
# found at http://waps.cfa.harvard.edu/MIST/model_grids.html#eeps specifically
# http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS.txz
Z, Y = 0.0143, 0.2703
eeps_dir = this_dir / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS"
eep_files = eeps_dir.glob("**/*.eep")

print(f"About to (over)write to the default models file:\t'{MistStellarModels.default_data_file}'")
if input("Are you sure (yes or [NO])? ").lower() in ["y", "yes"]:

    # This does the heavy lifting of parsing the source data for what's needed
    # and then writing (with overwrite) to the default data file.
    out_data_file = MistStellarModels.create_model_data_file(eep_files)

    # Now test the newly created model file with some well known lookups
    # This tests that it's accessible via the base's factory method
    lookup = MistStellarModels()

    print("\nSummary")
    print(f"A total of {lookup.row_count} data row(s).")
    print(f"Z values: {lookup.list_distinct_z_values()}")
    print(f"Y values: {lookup.list_distinct_y_values()}")
    print(f"Initial masses: {lookup.list_distinct_initial_masses()}")

    # log10(4.6 Gyr)~9.7 and log10(1.7 Gyr)~9.2
    for (init_mass, age) in [
        (1.00 * u.solMass, 9.7 * u.dex(u.yr)),
        (1.54 * u.solMass, 9.2 * u.dex(u.yr)),
        (1.3 * u.solMass, 9.2 * u.dex(u.yr))]:

        mass, radius, t_eff, lum, logg = lookup.lookup_stellar_parameters(Z, Y, init_mass, age)
        print(f"lookup_stellar_parameters({Z}, {Y}, {init_mass}, {age}):",
              f"M={mass:.3f}, R={radius:.3f}, TE={t_eff:.3f}, L={lum:.3f}, log(g)={logg:.3f}")

else:
    print("Ingest aborted.")
