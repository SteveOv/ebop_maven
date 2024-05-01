""" Utility functions for Stellar relations. """
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity, Dex, quantity_input

# This pylint disable makes it overlook its issue with astropy const aliases
# pylint: disable=no-member
@quantity_input(mass=u.kg, radius=u.m)
def log_g(mass: Quantity, radius: Quantity) -> Dex:
    """ 
    Calculate log(g) at the surface of a body with the given mass and radius. 

    log(g) = log((Gm/r^2).cgs)

    :mass: the stellar mass
    :radius: the stellar radius
    :returns: the log(g) value as a Dex quantity in units of dex (cgs)
    """
    return u.Dex(np.divide(np.multiply(const.G, mass.to(u.solMass)),
                           np.power(radius.to(u.solRad), 2)).cgs)
