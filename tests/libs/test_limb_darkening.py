""" Unit tests for the limb_darkening module. """
import unittest
import astropy.units as u
from astropy.units import UnitsError

from ebop_maven.libs.limb_darkening import lookup_tess_quad_ld_coeffs, lookup_tess_pow2_ld_coeffs

# pylint: disable=too-many-public-methods
class Testlimbdarkening(unittest.TestCase):
    """ Unit tests for the limb_darkening module. """

    # Expected lookup results
    # Copied directly from J_A+A_618_A20/table5.dat (a, b): a ~ col 28, b ~ col 36
    quad_coeffs_4_6500 = (0.3287, 0.2160)               # line 330
    quad_coeffs_4_7200 = (0.2821, 0.2295)               # line 378

    # Used to test around the gap in the data at 5000 K
    quad_coeffs_5_4900 = (0.4574, 0.1808)               # line 212
    quad_coeffs_5_5100 = (0.4291, 0.1908)               # line 220

    quad_coeffs_4_min = (0.4840, 0.3958)                # line 4
    quad_coeffs_4_max = (0.1278, 0.2553)                # line 570

    # Copied directly from J_A+A_674_A63/table1.dat (g, h): g ~ col 80, h ~ col 160
    pow2_coeffs_4_6500 = (0.60527825, 0.63326924)       # line 523
    pow2_coeffs_4_7200 = (0.58643009, 0.57964819)       # line 588

    # Used to test around the gap in the data at 5000 K
    pow2_coeffs_5_4900 = (0.68830452, 0.73045832)       # line 347
    pow2_coeffs_5_5100 = (0.67485744, 0.70541400)       # line 360

    pow2_coeffs_4_min = (0.96034133, 0.61861196)        # line 9
    pow2_coeffs_4_max = (0.57706984, 0.33700502)        # line 819

    #
    # TESS Quadratic LD Coeffs: lookup_tess_quad_ld_coeffs(logg, T_eff)
    #
    def test_lookup_tess_quad_ld_coeffs_arg_validation(self):
        """ Tests lookup_tess_quad_ld_coeffs(arg validation) raises Errors """
        self.assertRaises(UnitsError, lookup_tess_quad_ld_coeffs, 4. * u.W, 6500 * u.K)
        self.assertRaises(TypeError, lookup_tess_quad_ld_coeffs, 4., 6500 * u.K)
        self.assertRaises(TypeError, lookup_tess_quad_ld_coeffs, None, 6500 * u.K)
        self.assertRaises(UnitsError, lookup_tess_quad_ld_coeffs, 4. * u.dex, 6500 * u.deg)
        self.assertRaises(TypeError, lookup_tess_quad_ld_coeffs, 4. * u.dex, 6500)
        self.assertRaises(TypeError, lookup_tess_quad_ld_coeffs, 4. * u.dex, None)

    def test_lookup_tess_quad_ld_coeffs_logg_explicit_dex_units(self):
        """ Tests lookup_tess_quad_ld_coeffs(4. dex(cm/s^2), 6500 K) gets for (4. dex, 6500 K) """
        self.assertEqual(self.quad_coeffs_4_6500,
                         lookup_tess_quad_ld_coeffs(4. * u.dex(u.cm / u.s**2), 6500. * u.K))

    # Test finding correct data in the range where logg steps are 0.5 and teff are 100 K
    def test_lookup_tess_quad_ld_coeffs_exact_arguments(self):
        """ Tests lookup_tess_quad_ld_coeffs(4. dex, 6500. K) gets for (4. dex, 6500. K) """
        self.assertEqual(self.quad_coeffs_4_6500,
                         lookup_tess_quad_ld_coeffs(4. * u.dex, 6500. * u.K))

    def test_lookup_tess_quad_ld_coeffs_round_down_logg(self):
        """ Tests lookup_tess_quad_ld_coeffs(4.24 dex, 6500. K) gets for (4. dex, 6500. K) """
        self.assertEqual(self.quad_coeffs_4_6500,
                         lookup_tess_quad_ld_coeffs(4.24 * u.dex, 6500. * u.K))

    def test_lookup_tess_quad_ld_coeffs_round_up_logg(self):
        """ Tests lookup_tess_quad_ld_coeffs(3.75 dex, 6500. K) gets for (4. dex, 6500. K) """
        self.assertEqual(self.quad_coeffs_4_6500,
                         lookup_tess_quad_ld_coeffs(3.75 * u.dex, 6500. * u.K))

    def test_lookup_tess_quad_ld_coeffs_round_down_temp1(self):
        """ Tests lookup_tess_quad_ld_coeffs(4.0 dex, 6549. K) gets for (4. dex, 6500. K) """
        self.assertEqual(self.quad_coeffs_4_6500,
                         lookup_tess_quad_ld_coeffs(4. * u.dex, 6549. * u.K))

    def test_lookup_tess_quad_ld_coeffs_round_up_temp1(self):
        """ Tests lookup_tess_quad_ld_coeffs(4.0 dex, 6450. K) gets for (4. dex, 6500. K) """
        self.assertEqual(self.quad_coeffs_4_6500,
                         lookup_tess_quad_ld_coeffs(4. * u.dex, 6450. * u.K))

    # These 2 work in the region where the temp steps change from 100 to 200 K
    # They're both expected to return the values for logg==4.0, T_eff=7200.
    def test_lookup_tess_quad_ld_coeffs_round_down_temp2(self):
        """ Tests lookup_tess_quad_ld_coeffs(4.0 dex, 7299. K) gets for (4. dex, 7200. K) """
        self.assertEqual(self.quad_coeffs_4_7200,
                         lookup_tess_quad_ld_coeffs(4. * u.dex, 7299. * u.K))

    def test_lookup_tess_quad_ld_coeffs_round_up_temp2(self):
        """ Tests lookup_tess_quad_ld_coeffs(4.0 dex, 7100. K) gets for (4.0 dex, 7200. K) """
        self.assertEqual(self.quad_coeffs_4_7200,
                         lookup_tess_quad_ld_coeffs(4. * u.dex, 7100. * u.K))

    # There's a gap in the A&A, 618, A20, Table 5 quad coeffs between 4900 and 5100 K
    # Test the special case where we need to round away from 5000, otherwise we get a KeyError.
    def test_lookup_tess_quad_ld_coeffs_handles_gap_in_data(self):
        """ Tests lookup_tess_quad_ld_coeffs(near 5. dex, 5000 K) handles gap at 5000 K """
        self.assertEqual(self.quad_coeffs_5_4900,
                         lookup_tess_quad_ld_coeffs(4.81 * u.dex, 4956.7 * u.K))
        self.assertEqual(self.quad_coeffs_5_5100,
                         lookup_tess_quad_ld_coeffs(5.24 * u.dex, 5021.3 * u.K))

    # There's a gap in the A&A, 618, A20, Table 5 quad coeffs range from 2300 - 12000 K
    # Test the special cases where request outside this range -> expect to use limts
    def test_lookup_tess_quad_ld_coeffs_handles_temp_outside_range(self):
        """ Tests lookup_tess_quad_ld_coeffs(temp outside range) returns min or max """
        self.assertEqual(self.quad_coeffs_4_min,
                         lookup_tess_quad_ld_coeffs(4.03 * u.dex, 2000 * u.K))
        self.assertEqual(self.quad_coeffs_4_max,
                         lookup_tess_quad_ld_coeffs(4.03 * u.dex, 12600 * u.K))

    # These 2 test keys outside the known indices - rasing KeyError
    def test_lookup_tess_quad_ld_coeffs_unknown_logg(self):
        """ Tests lookup_tess_quad_ld_coeffs(7.2 dex, 6500. K) raises KeyError """
        self.assertRaises(KeyError, lookup_tess_quad_ld_coeffs, 7.2 * u.dex, 6500. * u.K)


    #
    # TESS power-2 LD Coeffs: lookup_tess_pow2_ld_coeffs(logg, T_eff)
    #
    def test_llookup_tess_pow2_ld_coeffs_arg_validation(self):
        """ Tests lookup_tess_pow2_ld_coeffs(arg validation) raises Errors """
        self.assertRaises(UnitsError, lookup_tess_pow2_ld_coeffs, 4. * u.W, 6500 * u.K)
        self.assertRaises(TypeError, lookup_tess_pow2_ld_coeffs, 4., 6500 * u.K)
        self.assertRaises(TypeError, lookup_tess_pow2_ld_coeffs, None, 6500 * u.K)
        self.assertRaises(UnitsError, lookup_tess_pow2_ld_coeffs, 4. * u.dex, 6500 * u.deg)
        self.assertRaises(TypeError, lookup_tess_pow2_ld_coeffs, 4. * u.dex, 6500)
        self.assertRaises(TypeError, lookup_tess_pow2_ld_coeffs, 4. * u.dex, None)

    def test_lookup_tess_pow2_ld_coeffs_explicit_dex_units(self):
        """ Tests lookup_tess_pow2_ld_coeffs(4. dex(cm/s^2), 6500 K) gets for (4. dex, 6500 K) """
        self.assertEqual(self.pow2_coeffs_4_6500,
                         lookup_tess_pow2_ld_coeffs(4. * u.dex(u.cm / u.s**2), 6500. * u.K))

    # Test finding correct data in the range where logg steps are 0.5 and teff are 100 K
    def test_lookup_tess_pow2_ld_coeffs_exact_arguments(self):
        """ Tests lookup_tess_pow2_ld_coeffs(4.0 dex, 6500. K) gets for (4.0 dex, 6500. K) """
        self.assertEqual(self.pow2_coeffs_4_6500,
                         lookup_tess_pow2_ld_coeffs(4. * u.dex, 6500. * u.K))

    def test_lookup_tess_pow2_ld_coeffs_round_down_logg(self):
        """ Tests lookup_tess_pow2_ld_coeffs(4.24 dex, 6500. K) gets for (4.0 dex, 6500. K) """
        self.assertEqual(self.pow2_coeffs_4_6500,
                         lookup_tess_pow2_ld_coeffs(4.24 * u.dex, 6500. * u.K))

    def test_lookup_tess_pow2_ld_coeffs_round_up_logg(self):
        """ Tests lookup_tess_pow2_ld_coeffs(3.75 dex, 6500. K) gets for (4.0 dex, 6500. K) """
        self.assertEqual(self.pow2_coeffs_4_6500,
                         lookup_tess_pow2_ld_coeffs(3.75 * u.dex, 6500. * u.K))

    def test_lookup_tess_pow2_ld_coeffs_round_down_temp1(self):
        """ Tests lookup_tess_pow2_ld_coeffs(4.0 dex, 6549. K) gets for (4.0 dex, 6500. K) """
        self.assertEqual(self.pow2_coeffs_4_6500,
                         lookup_tess_pow2_ld_coeffs(4. * u.dex, 6549. * u.K))

    def test_lookup_tess_pow2_ld_coeffs_round_up_temp1(self):
        """ Tests lookup_tess_pow2_ld_coeffs(4.0 dex, 6450. K) gets for (4.0 dex, 6500. K) """
        self.assertEqual(self.pow2_coeffs_4_6500,
                         lookup_tess_pow2_ld_coeffs(4. * u.dex, 6450. * u.K))

    # These 2 work in the region where the temp steps change from 100 to 200 K
    # They're both expected to return the values for logg==4.0, T_eff=7200.
    def test_lookup_tess_pow2_ld_coeffs_round_down_temp2(self):
        """ Tests lookup_tess_pow2_ld_coeffs(4.0 dex, 7299. K) gets for (4.0 dex, 7200. K) """
        self.assertEqual(self.pow2_coeffs_4_7200,
                         lookup_tess_pow2_ld_coeffs(4. * u.dex, 7299. * u.K))

    def test_lookup_tess_pow2_ld_coeffs_round_up_temp2(self):
        """ Tests lookup_tess_pow2_ld_coeffs(4.0 dex, 7100. K) gets for (4.0 dex, 7200. K) """
        self.assertEqual(self.pow2_coeffs_4_7200,
                         lookup_tess_pow2_ld_coeffs(4. * u.dex, 7100. * u.K))

    # There's a gap in the A&A, 664, A63, Table 1 pow2 coeffs between 4900 and 5100 K
    # Test the special case where we need to round away from 5000, otherwise we get a KeyError.
    def test_lookup_tess_pow2_ld_coeffs_handles_gap_in_data(self):
        """ Tests lookup_tess_pow2_ld_coeffs(near 5. dex, 5000 K) handles gap at 5000 K """
        self.assertEqual(self.pow2_coeffs_5_4900,
                         lookup_tess_pow2_ld_coeffs(4.96 * u.dex, 4996.7 * u.K))
        self.assertEqual(self.pow2_coeffs_5_5100,
                         lookup_tess_pow2_ld_coeffs(5.03 * u.dex, 5011.1 * u.K))

    # There's a gap in the A&A, 664, A63, Table 1 pow2 coeffs covers 2300 to 12000 K
    # Test the special case where we get a request outside this range -> expected to return limit.
    def test_lookup_tess_pow2_ld_coeffs_handles_temp_outside_range(self):
        """ Tests lookup_tess_pow2_ld_coeffs(temp outside range) returns min or max """
        self.assertEqual(self.pow2_coeffs_4_min,
                         lookup_tess_pow2_ld_coeffs(4.06 * u.dex, 2000 * u.K))
        self.assertEqual(self.pow2_coeffs_4_max,
                         lookup_tess_pow2_ld_coeffs(4.03 * u.dex, 12600 * u.K))

    # These 2 test keys outside the known indices - rasing KeyError
    def test_lookup_tess_pow2_ld_coeffs_unknown_logg(self):
        """ Tests lookup_tess_pow2_ld_coeffs(7.2 dex, 6500. K) raises KeyError """
        self.assertRaises(KeyError, lookup_tess_pow2_ld_coeffs, 7.2 * u.dex, 6500. * u.K)


if __name__ == "__main__":
    unittest.main()
