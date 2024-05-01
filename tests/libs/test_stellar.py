""" Unit tests for the stellar module. """
import unittest
import astropy.units as u
from astropy.units.core import UnitsError

from ebop_maven.libs.stellar import log_g

# pylint: disable=too-many-public-methods
class Teststellar(unittest.TestCase):
    """ Unit tests for the stellar module. """

    #
    # Test log_g(mass, radius)
    #
    def test_log_g_invalid_args(self):
        """ Tests log_g(invalid args) raises appropriate error """
        self.assertRaises(UnitsError, log_g, 1.57 * u.m, 2.11 * u.solRad)
        self.assertRaises(UnitsError, log_g, 1.57 * u.solMass, 2.11 * u.kg)
        self.assertRaises(TypeError, log_g, 1.57, 2.11 * u.solRad)
        self.assertRaises(TypeError, log_g, 1.57 * u.solMass, 2.11)
        self.assertRaises(TypeError, log_g, None, 2.11 * u.solRad)
        self.assertRaises(TypeError, log_g, 1.57 * u.solMass, None)

    def test_log_g_sol_si_values(self):
        """ Tests log_g(Sol SI values) ~= 4.4 """
        logg = log_g(1.99e30 * u.kg, 6.96e8 * u.m)
        self.assertAlmostEqual(4.4, logg.value, 1)

    def test_log_g_sol_fiducial_values(self):
        """ Tests log_g(Sol fiducial values) ~= 4.4 """
        logg = log_g(1* u.solMass, 1 * u.solRad)
        self.assertAlmostEqual(4.4, logg.value, 1)

    def test_log_g_cweri_fiducial_values(self):
        """ Tests log_g(CW Eri fiducial values) ~= 4.0 """
        logg = log_g(1.57 * u.solMass, 2.11 * u.solRad)
        self.assertAlmostEqual(4.0, logg.value, 1)

if __name__ == "__main__":
    unittest.main()
