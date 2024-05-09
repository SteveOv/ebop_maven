""" Utility functions for Orbital relations. """
from typing import Union, Tuple
from enum import Flag
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity, quantity_input

# This pylint disable overlooks the issue it has with astropy const aliases
# pylint: disable=no-member
# pylint: disable=too-many-arguments

__four_pi_squared = np.multiply(4, np.power(np.pi, 2))

class EclipseType(Flag):
    """ Indicates which type of eclipse """
    PRIMARY = 1
    SECONDARY = 2
    BOTH = 3

@quantity_input(m1=u.kg, m2=u.kg, a=u.m)
def orbital_period(m1: Quantity, m2: Quantity, a: Quantity) -> u.d:
    """
    Calculates the orbital period from the two components' masses and the
    orbital semi-major axis. The calculation is based on Kepler's 3rd Law.

    P^2 = (4π^2*a^3)/G(m1+m2)

    :m1: the first mass
    :m2: the second mass
    :a: the semi-major axis length
    :returns: the orbital period in units of d
    """
    p_squared = np.divide(np.multiply(__four_pi_squared, np.power(a, 3)),
                          np.multiply(const.G, np.add(m1, m2)))
    return np.sqrt(p_squared)

@quantity_input(m1=u.kg, m2=u.kg, period=u.s)
def semi_major_axis(m1: Quantity, m2: Quantity, period: Quantity) -> u.solRad:
    """
    Calculates the orbital semi-major axis of two orbital components from their
    masses and orbital period. The calculation is based on Kepler's 3rd law.

    a^3 = G(m1+m2)P^2/4π^2

    :m1: the first mass
    :m2: the second mass
    :period: the components' orbital period
    :returns: the semi-major axis in units of solRad
    """
    a_cubed = np.divide(np.multiply(const.G, np.multiply(np.add(m1, m2), np.power(period, 2))),
                        __four_pi_squared)
    return np.cbrt(a_cubed)

@quantity_input(inc=u.deg, omega=u.deg)
def impact_parameter(r1: float,
                     inc: Quantity,
                     e: float,
                     omega: Quantity=None,
                     esinw: float=None,
                     eclipse: EclipseType=EclipseType.PRIMARY) \
                        -> Union[float, Tuple[float, float]]:
    """
    Calculate the impact parameter of one or both eclipse types. This uses
    the primary star's fractional radius, orbital inclination, eccentricity
    and one of either omega or esinw [e*sin(omega)]. While the function can
    calculate e*sin(omega) from e and omega the function may be quicker if
    you supply it directly (as esinw) if you already have it.

    :r1: fractional radius of the primary star
    :inc: the orbital inclination
    :e: the orbital eccentricity
    :omega: the argument of periastron
    :esinw: the e*sin(omega) Poincare element
    :eclipse: the type of eclipse to calculate for PRIMARY, SECONDARY or BOTH.
    :returns: either a single value where type is PRIMARY or SECONDARY or a
    tuple holding both values where type is BOTH.
    """
    # Primary eclipse:      bA = (1/r1) * cos(inc) * (1-e^2 / 1+esinw)
    # Secondary eclipse:    bB = (1/r1) * cos(inc) * (1-e^2 / 1-esinw)
    # Only difference is the final divisor so work the common dividend out
    dividend = np.multiply(np.reciprocal(r1),
                           np.multiply(np.cos(inc.to(u.rad).value),
                                       np.subtract(1., np.power(e, 2))))

    # Must have either esinw or omega with esinw taking precedent if both given.
    if esinw is None:
        if omega is None:
            raise ValueError("Must provide a value for either omega or esinw")
        esinw = np.multiply(e, np.sin(omega.to(u.rad).value))

    if eclipse & EclipseType.PRIMARY:
        bp = np.divide(dividend, np.add(1, esinw))
    if eclipse & EclipseType.SECONDARY:
        bs = np.divide(dividend, np.subtract(1, esinw))

    return (bp, bs) if eclipse == EclipseType.BOTH \
                else bp if eclipse == EclipseType.PRIMARY else bs

@quantity_input(omega=u.deg)
def orbital_inclination(r1: float,
                        b: float,
                        e: float,
                        omega: Quantity=None,
                        esinw: float=None,
                        eclipse: EclipseType=EclipseType.PRIMARY)\
                            -> u.deg:
    """
    Calculate the orbital inclination from an impact parameter. This uses
    the primary star's fractional radius, orbital eccentricity and one of either
    omega or esinw [e*sin(omega)] along with the supplied impact parameter.

    :r1: fractional radius of the primary star
    :b: the chosen impact parameter
    :e: the orbital eccentricity
    :omega: the argument of periastron
    :esinw: the e*sin(omega) Poincare element
    :eclipse: the type of eclipse the impact parameter is from; PRIMARY or SECONDARY (not BOTH).
    :returns: The inclination as a Quantity in degrees
    """
    # From primary eclipse/impact param:  i = arccos(bP * r1 * (1+esinw)/(1-e^2))
    # From secodary eclipse/impact param: i = arccos(bS * r1 * (1-esinw)/(1-e^2))

    # Must have either esinw or omega with esinw taking precedent if both given.
    if esinw is None:
        if omega is None:
            raise ValueError("Must provide a value for either omega or esinw")
        esinw = np.multiply(e, np.sin(omega.to(u.rad).value))

    if eclipse == EclipseType.PRIMARY:
        dividend = np.add(1, esinw)
    elif eclipse == EclipseType.SECONDARY:
        dividend = np.subtract(1, esinw)
    else:
        raise ValueError(f"{EclipseType.BOTH} is not supported")
    eccentricity_factor = np.divide(dividend, np.subtract(1, np.power(e, 2)))
    return np.rad2deg(np.arccos(np.multiply(np.multiply(b, r1), eccentricity_factor))) * u.deg
