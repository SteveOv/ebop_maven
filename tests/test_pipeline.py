""" Unit tests for the pipeline module. """
# pylint: disable=no-member
import unittest

import astropy.units as u
from astropy.time import TimeDelta
import tests.helpers.lightcurve_helpers as th

from traininglib import pipeline

# pylint: disable=too-many-public-methods, line-too-long
class Testpipeline(unittest.TestCase):
    """ Unit tests for the pipeline module. """

    #
    #   TESTS find_lightcurve_segments(lc, threshold) -> Generator[(first, last)]
    #
    def test_find_lightcurve_segments_cw_eri_return_indices(self):
        """ Tests find_lightcurve_segments(CW Eri sector 31, th=0.5 d, times=False) -> 2 segments """
        lc = th.load_lightcurve("CW Eri")
        segmment_ixs = list(pipeline.find_lightcurve_segments(lc, TimeDelta(0.5 * u.d), False))
        self.assertEqual(len(segmment_ixs), 2)
        for seg_ixs, exp_time_values in zip(segmment_ixs, [(2144.5, 2156.7), (2158.9, 2169.9)]):
            self.assertAlmostEqual(lc[seg_ixs[0]]["time"].value, exp_time_values[0], 1)
            self.assertAlmostEqual(lc[seg_ixs[1]]["time"].value, exp_time_values[1], 1)

    def test_find_lightcurve_segments_cw_eri_return_times(self):
        """ Tests find_lightcurve_segments(CW Eri sector 31, th=0.5 d, times=True) -> 2 segments """
        lc = th.load_lightcurve("CW Eri")
        segmment_times = list(pipeline.find_lightcurve_segments(lc, TimeDelta(0.5 * u.d), True))
        self.assertEqual(len(segmment_times), 2)
        for seg_times, exp_time_values in zip(segmment_times, [(2144.5, 2156.7), (2158.9, 2169.9)]):
            self.assertAlmostEqual(seg_times[0].value, exp_time_values[0], 1)
            self.assertAlmostEqual(seg_times[1].value, exp_time_values[1], 1)

    def test_find_lightcurve_segments_v889_aql(self):
        """ Tests find_lightcurve_segments(V889 Aql sector 40, th=0.5 d) -> 2 segments """
        # Tougher test than CW Eri as lower cadence and a smaller gap
        lc = th.load_lightcurve("V889 Aql")
        segment_times = list(pipeline.find_lightcurve_segments(lc, TimeDelta(0.5 * u.d), True))
        self.assertEqual(len(segment_times), 2)
        for seg_times, exp_time_values in zip(segment_times, [(2390.7, 2404.4), (2405.3, 2418.9)]):
            self.assertAlmostEqual(seg_times[0].value, exp_time_values[0], 1)
            self.assertAlmostEqual(seg_times[1].value, exp_time_values[1], 1)

    def test_find_lightcurve_segments_v889_aql_high_threshold(self):
        """ Tests find_lightcurve_segments(V889 Aql sector 40, th=1 d) -> 1 segment (whole LC) """
        # V889 Aql mid-sector interval is ~0.9 d, so should not be split when threshold is 1 d
        lc = th.load_lightcurve("V889 Aql")
        segment_times = list(pipeline.find_lightcurve_segments(lc, TimeDelta(1.0 * u.d), True))
        self.assertEqual(len(segment_times), 1)
        for seg_times, exp_time_values in zip(segment_times, [(2390.7, 2418.9)]):
            self.assertAlmostEqual(seg_times[0].value, exp_time_values[0], 1)
            self.assertAlmostEqual(seg_times[1].value, exp_time_values[1], 1)

if __name__ == "__main__":
    unittest.main()
