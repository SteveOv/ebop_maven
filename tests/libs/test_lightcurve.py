""" Unit tests for the lightcurve module. """
import unittest

import astropy.units as u
from astropy.time import TimeDelta
import tests.libs.helpers.lightcurve_helpers as th

from ebop_maven.libs import lightcurve

# pylint: disable=too-many-public-methods, line-too-long
class Testlightcurve(unittest.TestCase):
    """ Unit tests for the lightcurve module. """
    #
    #   TEST expected_secondary_phase(esinw, ecc) -> float
    #
    def test_expected_secondary_phase_gg_lup(self):
        """ Tests expected_secondary_phase(values for GG Lup from Hilditch pp238) """
        ecosw = 0.0098
        esinw = 0.15
        ecc = 0.15
        phi_s = lightcurve.expected_secondary_phase(ecosw, ecc=ecc)
        self.assertAlmostEqual(phi_s, 0.5064, 3)
        phi_s = lightcurve.expected_secondary_phase(ecosw, esinw=esinw)
        self.assertAlmostEqual(phi_s, 0.5064, 3)

    def test_expected_secondary_phase_cw_eri(self):
        """ Tests expected_secondary_phase(values for CW Eri from Overall & Southworth 2024) """
        target = th.KNOWN_TARGETS["CW Eri"]

        phi_s = lightcurve.expected_secondary_phase(target.ecosw, target.ecc)
        self.assertAlmostEqual(phi_s, target.expect_phase2, 3)

        phi_s = lightcurve.expected_secondary_phase(target.ecosw, esinw=target.esinw)
        self.assertAlmostEqual(phi_s, target.expect_phase2, 3)

    #
    #   TESTS expected_ratio_of_eclipse_duration(esinw) -> float
    #
    def test_expected_ratio_of_eclipse_duration_gg_lup(self):
        """ Tests expected_ratio_of_eclipse_duration(values for GG Lup from Hilditch) """
        esinw = 0.15
        ds_over_dp = lightcurve.expected_ratio_of_eclipse_duration(esinw)
        self.assertAlmostEqual(ds_over_dp, 1.353, 3)

    def test_expected_ratio_of_eclipse_duration_cw_eri(self):
        """ Tests expected_ratio_of_eclipse_duration(values for CW Eri from Overall & Southworth 2024) """
        target = th.KNOWN_TARGETS["CW Eri"]
        ds_over_dp = lightcurve.expected_ratio_of_eclipse_duration(target.esinw)
        self.assertAlmostEqual(ds_over_dp, target.expect_width2, 3)

    #
    #   TESTS find_lightcurve_segments(lc, threshold) -> Generator[(first, last)]
    #
    def test_find_lightcurve_segments_cw_eri_return_indices(self):
        """ Tests find_lightcurve_segments(CW Eri sector 31, th=0.5 d, times=False) -> 2 segments """
        lc = th.load_lightcurve("CW Eri")
        segmment_ixs = list(lightcurve.find_lightcurve_segments(lc, TimeDelta(0.5 * u.d), False))
        self.assertEqual(len(segmment_ixs), 2)
        for seg_ixs, exp_time_values in zip(segmment_ixs, [(2144.5, 2156.7), (2158.9, 2169.9)]):
            self.assertAlmostEqual(lc[seg_ixs[0]]["time"].value, exp_time_values[0], 1)
            self.assertAlmostEqual(lc[seg_ixs[1]]["time"].value, exp_time_values[1], 1)

    def test_find_lightcurve_segments_cw_eri_return_times(self):
        """ Tests find_lightcurve_segments(CW Eri sector 31, th=0.5 d, times=True) -> 2 segments """
        lc = th.load_lightcurve("CW Eri")
        segmment_times = list(lightcurve.find_lightcurve_segments(lc, TimeDelta(0.5 * u.d), True))
        self.assertEqual(len(segmment_times), 2)
        for seg_times, exp_time_values in zip(segmment_times, [(2144.5, 2156.7), (2158.9, 2169.9)]):
            self.assertAlmostEqual(seg_times[0].value, exp_time_values[0], 1)
            self.assertAlmostEqual(seg_times[1].value, exp_time_values[1], 1)

    def test_find_lightcurve_segments_v889_aql(self):
        """ Tests find_lightcurve_segments(V889 Aql sector 40, th=0.5 d) -> 2 segments """
        # Tougher test than CW Eri as lower cadence and a smaller gap
        lc = th.load_lightcurve("V889 Aql")
        segment_times = list(lightcurve.find_lightcurve_segments(lc, TimeDelta(0.5 * u.d), True))
        self.assertEqual(len(segment_times), 2)
        for seg_times, exp_time_values in zip(segment_times, [(2390.7, 2404.4), (2405.3, 2418.9)]):
            self.assertAlmostEqual(seg_times[0].value, exp_time_values[0], 1)
            self.assertAlmostEqual(seg_times[1].value, exp_time_values[1], 1)

    def test_find_lightcurve_segments_v889_aql_high_threshold(self):
        """ Tests find_lightcurve_segments(V889 Aql sector 40, th=1 d) -> 1 segment (whole LC) """
        # V889 Aql mid-sector interval is ~0.9 d, so should not be split when threshold is 1 d
        lc = th.load_lightcurve("V889 Aql")
        segment_times = list(lightcurve.find_lightcurve_segments(lc, TimeDelta(1.0 * u.d), True))
        self.assertEqual(len(segment_times), 1)
        for seg_times, exp_time_values in zip(segment_times, [(2390.7, 2418.9)]):
            self.assertAlmostEqual(seg_times[0].value, exp_time_values[0], 1)
            self.assertAlmostEqual(seg_times[1].value, exp_time_values[1], 1)

if __name__ == "__main__":
    unittest.main()
