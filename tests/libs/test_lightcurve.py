""" Unit tests for the lightcurve module. """
import unittest

import tests.libs.helpers.lightcurve_helpers as th

from ebop_maven.libs.lightcurve import expected_secondary_phase, expected_ratio_of_eclipse_duration

# pylint: disable=too-many-public-methods, line-too-long
class Testlightcurve(unittest.TestCase):
    """ Unit tests for the lightcurve module. """
    #
    #   TEST expected_secondary_phase(esinw, ecc) -> float
    #
    def test_expected_secondary_phase_gg_lup(self):
        """ Tests expected_secondary_phase(values for GG Lup from Hilditch) """
        ecosw = 0.0098
        ecc = 0.15
        phi_s = expected_secondary_phase(ecosw, ecc)
        self.assertAlmostEqual(phi_s, 0.5064, 3)

    def test_expected_secondary_phase_cw_eri(self):
        """ Tests expected_secondary_phase(values for CW Eri from Overall & Southworth 2024) """
        target = th.KNOWN_TARGETS["CW Eri"]
        phi_s = expected_secondary_phase(target.ecosw, target.ecc)
        self.assertAlmostEqual(phi_s, target.expect_phase2, 3)

    #
    #   TESTS expected_ratio_of_eclipse_duration(esinw) -> float
    #
    def test_expected_ratio_of_eclipse_duration_gg_lup(self):
        """ Tests expected_ratio_of_eclipse_duration(values for GG Lup from Hilditch) """
        esinw = 0.15
        ds_over_dp = expected_ratio_of_eclipse_duration(esinw)
        self.assertAlmostEqual(ds_over_dp, 1.353, 3)

    def test_expected_ratio_of_eclipse_duration_cw_eri(self):
        """ Tests expected_ratio_of_eclipse_duration(values for CW Eri from Overall & Southworth 2024) """
        target = th.KNOWN_TARGETS["CW Eri"]
        ds_over_dp = expected_ratio_of_eclipse_duration(target.esinw)
        self.assertAlmostEqual(ds_over_dp, target.expect_width2, 3)


if __name__ == "__main__":
    unittest.main()
