""" Module to ingest parsec evo tracks, interpolate and write to models.pkl.xz file """
from inspect import getsourcefile
from pathlib import Path
import sys

import numpy as np
import astropy.units as u

# Hack so this script imports Parsec class from separate /code folder structure.
# pylint: disable=import-error, wrong-import-position
this_dir = Path(getsourcefile(lambda:0)).parent
sys.path.append(f"{(this_dir / '../../..').resolve()}")
from parsec import Parsec

print(f"\nAbout to ingest PARSEC data from:\t'{this_dir}'.")
print(f"To overwrite the default models file:\t'{Parsec.default_data_file}'")
if input("Are you sure (yes or [NO])? ").lower() in ["y", "yes"]:

    print("\nIngesting.")
    age_step = Parsec.default_age_step.to(u.Gyr)
    Parsec.ingest_parsec_tracks(source_folder=this_dir, age_step=age_step)

    print("\nPerforming test lookup on newly created data file.")
    ps = Parsec()
    for (init_m, age, z, y) in [
        (1. * u.solMass, 4.6 * u.Gyr, 0.014, 0.273),
        (1.55 * u.solMass, 1.7 * u.Gyr, 0.017, 0.279),
        (1.3 * u.solMass, 1.7 * u.Gyr, 0.017, 0.279)]:
        mass, radius, t_eff, lum = \
            ps.lookup_stellar_parameters(init_m, age, z, y, is_hb=False)
        print(f"lookup_stellar_parameters(...): "\
            f"MASS={mass:.3f}, R={radius:.3f}, TE={t_eff:.3f}, L={lum:.3f}")

    print("\nSummary")
    print(f"Y values: {ps.list_distinct_y_values()}")
    print(f"Z values: {ps.list_distinct_z_values()}")
    print(f"Initial masses: {ps.list_distinct_initial_masses()}")

    ages = ps.list_distinct_ages().to(u.Gyr)
    print(f"Ages: from {np.min(ages)} to {np.max(ages)} in steps of {age_step}")

else:
    print("Ingest aborted.")
