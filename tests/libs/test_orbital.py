""" Unit tests for the orbital module. """
import unittest
import astropy.units as u
from astropy.units.core import UnitsError

from ebop_maven.libs.orbital import orbital_period, semi_major_axis, impact_parameter
from ebop_maven.libs.orbital import EclipseType

# pylint: disable=too-many-public-methods, no-member, line-too-long
class Testorbital(unittest.TestCase):
    """ Unit tests for the orbital module. """
    SOL_MASS_SI = (1 * u.solMass).to(u.kg)
    EARTH_MASS_SI = (1 * u.earthMass).to(u.kg)
    YEAR_SI = (1 * u.yr).to(u.s)
    AU_SI = (1 * u.au).to(u.m)

    # Both eclipse types are partial (bA=0.36, bB=0.35 from fitting)
    _cw_eri = { "r1": .180, "inc": 86.38 * u.deg, "e": 0.0131, "omega": 292.4 * u.deg }

    # Both eclipse types are partial (bA = bB = 1.3 from fitting)
    # More inclined than CW Eri and non eccentric
    _v570_per = { "r1": 0.169, "inc": 77.3 * u.deg, "e": 0, "omega": 0 * u.deg}

    # Both eclipses are total and near edge on (bA = bB = 0.056 from fitting)
    _zeta_phe = { "r1": 0.258, "inc": 89.14 * u.deg, "e": 0.0116, "omega": 307 * u.deg}

    # Synthetic system where only primary eclipses occur (highly eccentric)
    _prim_only = { "r1": 1/10, "inc": 80. * u.deg, "e": 0.3, "omega": 90 * u.deg}

    # Fractional radius of the Earth in Earth/Sol system
    _r1_sol_earth = (1. * u.solRad / (1. * u.au).to(u.solRad)).value

    #
    # Test orbital_period(m1, m2, a)
    #
    def test_orbital_period_args_invalid_units(self):
        """ Tests orbital_period(args with invalid units) -> UnitsError """
        self.assertRaises(UnitsError, orbital_period, 1 * u.solRad, 1 * u.earthMass, 1 * u.au)
        self.assertRaises(UnitsError, orbital_period, 1 * u.solMass, 1 * u.earthRad, 1 * u.au)
        self.assertRaises(UnitsError, orbital_period, 1 * u.solMass, 1 * u.earthMass, 1 * u.deg)

    def test_orbital_period_args_no_units(self):
        """ Tests orbital_period(args without units) -> TypeError """
        self.assertRaises(TypeError, orbital_period, 1, 1 * u.earthMass, 1 * u.au)
        self.assertRaises(TypeError, orbital_period, 1 * u.earthMass, 1, 1 * u.au)
        self.assertRaises(TypeError, orbital_period, 1 * u.earthMass, 1 * u.earthMass, 1)

    def test_orbital_period_args_none(self):
        """ Tests orbital_period(args set to None) -> TypeError """
        self.assertRaises(TypeError, orbital_period, None, 1 * u.earthMass, 1 * u.au)
        self.assertRaises(TypeError, orbital_period, 1 * u.earthMass, None, 1 * u.au)
        self.assertRaises(TypeError, orbital_period, 1 * u.earthMass, 1 * u.earthMass, None)

    def test_orbital_period_sol_fiducial_values(self):
        """ Tests orbital_period(Sol/Earth fiducial values) ~= 365.25 d """
        period = orbital_period(1 * u.solMass, 1 * u.earthMass, 1 * u.au)
        self.assertAlmostEqual(365.26, period.value, 2)

    def test_orbital_period_sol_si_values(self):
        """ Tests orbital_period(Sol/Earth SI values) ~= 365.25 d """
        period = orbital_period(self.SOL_MASS_SI, self.EARTH_MASS_SI, self.AU_SI)
        self.assertAlmostEqual(365.26, period.value, 2)

    def test_orbital_period_cweri_fiducial_values(self):
        """ Tests orbital_period(CW Eri fiducial values) ~= 2.73 d """
        period = orbital_period(1.57 * u.solMass, 1.31 * u.solMass, 11.69 * u.solRad)
        self.assertAlmostEqual(2.73, period.value, 2)

    #
    # Test semi_major_axis(m1, m2, period)
    #
    def test_semi_major_axis_args_invalid_units(self):
        """ Tests semi_major_axis(args with invalid units) -> UnitsError """
        self.assertRaises(UnitsError, semi_major_axis, 1 * u.solRad, 1 * u.earthMass, 1 * u.au)
        self.assertRaises(UnitsError, semi_major_axis, 1 * u.solMass, 1 * u.deg, 1 * u.au)
        self.assertRaises(UnitsError, semi_major_axis, 1 * u.solMass, 1 * u.solMass, 1 * u.deg)

    def test_semi_major_axis_args_no_units(self):
        """ Tests semi_major_axis(args without units) -> TypeError """
        self.assertRaises(TypeError, semi_major_axis, 1, 1 * u.earthMass, 1 * u.au)
        self.assertRaises(TypeError, semi_major_axis, 1 * u.solMass, 1, 1 * u.au)
        self.assertRaises(TypeError, semi_major_axis, 1 * u.solMass, 1 * u.solMass, 1)

    def test_semi_major_axis_args_none(self):
        """ Tests semi_major_axis(args set to None) -> TypeError """
        self.assertRaises(TypeError, semi_major_axis, None, 1 * u.earthMass, 1 * u.au)
        self.assertRaises(TypeError, semi_major_axis, 1 * u.solMass, None, 1 * u.au)
        self.assertRaises(TypeError, semi_major_axis, 1 * u.solMass, 1 * u.solMass, None)

    def test_semi_major_axis_sol_fiducial_values(self):
        """ Tests semi_major_axis(Sol/Earth fiducial values) ~= 1 AU """
        a = semi_major_axis(1 * u.solMass, 1 * u.earthMass, 1 * u.yr)
        self.assertAlmostEqual(1.00, a.to(u.au).value, 2)

    def test_semi_major_axis_sol_si_values(self):
        """ Tests semi_major_axis(Sol/Earth si values) ~= 1 AU """
        a = semi_major_axis(self.SOL_MASS_SI, self.EARTH_MASS_SI, self.YEAR_SI)
        self.assertAlmostEqual(1.50e8, a.to(u.km).value, -6)

    def test_semi_major_axis_cweri_fiducial_values(self):
        """ Tests semi_major_axis(CW Eri fiducial values) ~= 11.69 solRad """
        a = semi_major_axis(1.57 * u.solMass, 1.31 * u.solMass, 2.728 * u.d)
        self.assertAlmostEqual(11.69, a.to(u.solRad).value, 2)

    #
    # Tests impact_parameter(rA, inc, e, [omega], [esinw], [eclipse=EclipseType])
    #
    def test_impact_parameter_args_invalid_units(self):
        """ Tests impact_parameter(invalid argument units) raises UnitsError """
        self.assertRaises(UnitsError, impact_parameter, 1., 90 * u.K,   0,      0 * u.rad)
        self.assertRaises(UnitsError, impact_parameter, 1., 90 * u.deg, 0,      0 * u.K)

    def test_impact_parameter_args_no_units(self):
        """ Tests impact_parameter(missing argument units) raises TypeError """
        self.assertRaises(TypeError, impact_parameter, 1.,  90,         0,      0 * u.rad)
        self.assertRaises(TypeError, impact_parameter, 1.,  90 * u.deg, 0,      0)

    def test_impact_parameter_args_none(self):
        """ Tests impact_parameter(RA, a, inc or e is None) raises TypeError """
        self.assertRaises(TypeError, impact_parameter, None, 90 * u.deg, 0,      0 * u.rad)
        self.assertRaises(TypeError, impact_parameter, 1.,   None,       0,      0 * u.rad)
        self.assertRaises(TypeError, impact_parameter, 1.,   90 * u.deg, None,   0 * u.rad)

    def test_impact_parameter_both_omega_and_esinw_none(self):
        """ Tests impact_parameter(both omega and esinw is None) raises ValueError """
        self.assertRaises(ValueError, impact_parameter, 0.1, 90 * u.deg, e=0, omega=None, esinw=None)

    def test_impact_parameter_esinw_none(self):
        """ Tests impact_parameter(esinw is None) given e and omega used to calc esinw """
        kwargs = { "r1": self._r1_sol_earth, "inc": 80. * u.deg }
        b = impact_parameter(**kwargs, e=0.5, omega=0 * u.deg, esinw=None, eclipse=EclipseType.PRIMARY)
        self.assertAlmostEqual(b, 28.005, 4, "b != 28.0050 (the value expected when esinw==0)")

    def test_impact_parameter_omega_none(self):
        """ Tests impact_parameter(omega is None) given esinw used """
        kwargs = { "r1": self._r1_sol_earth, "inc": 80. * u.deg }
        b = impact_parameter(**kwargs, e=0.5, omega=None, esinw=0., eclipse=EclipseType.PRIMARY)
        self.assertAlmostEqual(b, 28.005, 4, "b != 28.0050 (the value expected when esinw==0)")

    def test_impact_parameter_both_omega_and_esinw_given(self):
        """ Tests impact_parameter(both omega and esinw given) given esinw used """
        kwargs = { "r1": self._r1_sol_earth, "inc": 80. * u.deg }
        # With e=0.5 and omega=90 deg the calculated value of esinw=0.5, which yields b=18.6700.
        # By also specifying esinw=0, which yields b=28.0050, we can tell which has been used.
        b = impact_parameter(**kwargs, e=0.5, omega=90 * u.deg, esinw=0., eclipse=EclipseType.PRIMARY)
        self.assertAlmostEqual(b, 28.005, 4, "b != 28.0050 (the value expected when esinw==0)")

    def test_impact_parameter_circular_edge_on_primary_eclipse(self):
        """ Tests impact_parameter(primary of edge on orbit) -> b == 0 """
        kwargs = { "r1": self._r1_sol_earth, "inc": 90. * u.deg }
        b = impact_parameter(**kwargs, e=0, omega=0 * u.rad, esinw=None, eclipse=EclipseType.PRIMARY)
        self.assertAlmostEqual(b, 0., 12, "b != 0 when esinw is None")
        b = impact_parameter(**kwargs, e=0, omega=None, esinw=0, eclipse=EclipseType.PRIMARY)
        self.assertAlmostEqual(b, 0., 12, "b != 0 when esinw is not None")

    def test_impact_parameter_circular_edge_on_secondary_eclipse(self):
        """ Tests impact_parameter(secondary of edge on orbit) -> b == 0 """
        kwargs = { "r1": self._r1_sol_earth, "inc": 90. * u.deg }
        b = impact_parameter(**kwargs, e=0, omega=0 * u.rad, esinw=None, eclipse=EclipseType.SECONDARY)
        self.assertAlmostEqual(b, 0., 12, "b != 0 when esinw is None")
        b = impact_parameter(**kwargs, e=0, omega=None, esinw=0, eclipse=EclipseType.SECONDARY)
        self.assertAlmostEqual(b, 0., 12, "b != 0 when esinw is not None")

    def test_impact_parameter_circular_edge_on_both_eclipses(self):
        """ Tests impact_parameter(both eclipses of edge on orbit) -> (bp, bs) == (0, 0) """
        kwargs = { "r1": self._r1_sol_earth, "inc": 90. * u.deg }
        (bp, bs) = impact_parameter(**kwargs, e=0, omega=0 * u.rad, esinw=None, eclipse=EclipseType.BOTH)
        self.assertAlmostEqual(bp, 0., 12)
        self.assertAlmostEqual(bs, 0., 12)
        (bp, bs) = impact_parameter(**kwargs, e=0, omega=None, esinw=0, eclipse=EclipseType.BOTH)
        self.assertAlmostEqual(bp, 0., 12)
        self.assertAlmostEqual(bs, 0., 12)

    def test_impact_parameter_cw_eri_known_impact_parameters(self):
        """ Tests impact_parameter(CW Eri) -> known b values from fitting """
        kwargs = self._cw_eri.copy()
        exp_bp, exp_bs, places = 0.36, 0.35, 2
        # Separately
        self.assertAlmostEqual(exp_bp, impact_parameter(**kwargs, eclipse=EclipseType.PRIMARY), places)
        self.assertAlmostEqual(exp_bs, impact_parameter(**kwargs, eclipse=EclipseType.SECONDARY), places)
        # Both at once
        (bp, bs) = impact_parameter(**kwargs, eclipse=EclipseType.BOTH)
        self.assertAlmostEqual(exp_bp, bp, places)
        self.assertAlmostEqual(exp_bs, bs, places)

    def test_impact_parameter_v570_per_known_grazing_impact_parameters(self):
        """ Tests impact_parameter(V570 Per) -> known (grazing) b values from fitting """
        kwargs = self._v570_per.copy()
        self.assertAlmostEqual(1.3, impact_parameter(**kwargs, eclipse=EclipseType.PRIMARY), 1)
        self.assertAlmostEqual(1.3, impact_parameter(**kwargs, eclipse=EclipseType.SECONDARY), 1)

    def test_impact_parameter_zeta_phe_known_total_impact_parameters(self):
        """ Tests impact_parameter(Zeta Phe) -> known (total) b values from fitting """
        kwargs = self._zeta_phe.copy()
        self.assertAlmostEqual(0.06, impact_parameter(**kwargs, eclipse=EclipseType.PRIMARY), 2)
        self.assertAlmostEqual(0.06, impact_parameter(**kwargs, eclipse=EclipseType.SECONDARY), 2)

    def test_impact_parameter_highly_inclined_primary_eclipse(self):
        """ Tests impact_parameter(CW Eri like configuration) -> correct parameter """
        kwargs = self._cw_eri.copy()
        kwargs["inc"] = 45 * u.deg
        b = impact_parameter(**kwargs, eclipse=EclipseType.PRIMARY)
        self.assertTrue(b > 3.)

if __name__ == "__main__":
    unittest.main()
