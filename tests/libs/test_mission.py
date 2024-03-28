""" Unit tests for the Mission base class and sub classes. """
import unittest
import astropy.units as u

from ebop_maven.libs.mission import Mission, Tess, Kepler

# pylint: disable=too-many-public-methods, line-too-long
class TestMission(unittest.TestCase):
    """ Unit tests for the Mission base class and sub classes. """
    _all_missions = [Tess, Kepler]

    #
    # Tests Mission.get_instance(mission_name: str) -> Mission:
    #
    def test_mission_get_instance_unknown_missions(self):
        """ Tests the Mission get_instance() function with unknown missions. """
        self.assertRaises(KeyError, Mission.get_instance, "Test")
        self.assertRaises(KeyError, Mission.get_instance, "Mission-Impossible")
        self.assertRaises(KeyError, Mission.get_instance, "Mission-Critical")

    def test_mission_get_instance_known_missions(self):
        """ Tests the Mission get_instance() function with valid variations of known missions. """
        for mission_name, mission_type in [("Tess", Tess), ("tess", Tess),
                                           ("Kepler", Kepler), ("KEPLER  ", Kepler)]:
            mission = Mission.get_instance(mission_name)
            self.assertIsInstance(mission, mission_type)

    def test_mission_get_instance_assert_caching(self):
        """ Tests the Mission get_instance() to assert caching of instances """
        # Currently uses a simple lru_cache decorator
        # so it's dependent on consistent naming
        instance1 = Mission.get_instance("Tess")
        instance2 = Mission.get_instance("Tess")
        self.assertEqual(instance1, instance2)


    #
    # Tests base/sub-class get_response_function()
    #
    def test_tess_get_response_function(self):
        """ Tests the TESS get_response_function(). """
        rf = Tess.get_response_function()
        self.assertIsNotNone(rf)
        self.assertEqual(rf.loc[800]["coefficient"], 0.777)
        self.assertEqual(len(rf.loc[600:1000]), 201)

    def test_kepler_get_response_function(self):
        """ Tests the Kepler get_response_function(). """
        rf = Kepler.get_response_function()
        self.assertIsNotNone(rf)
        self.assertEqual(rf.loc[500]["coefficient"], 6.239e-1)
        self.assertEqual(len(rf.loc[400:900]), 501)

    def test_mission_get_response_function(self):
        """ Tests polymorphic use of Mission get_response_function(). """
        for mission in self._all_missions:
            rf = mission.get_response_function()
            self.assertIsNotNone(rf)
            self.assertEqual(len(rf.loc[700]), 1)

    def test_mission_get_response_function_response_caching(self):
        """ Tests Mission subclass get_response_function() response caching. """
        for mission in self._all_missions:
            rf1 = mission.get_response_function()
            rf2 = mission.get_response_function()
            self.assertTrue(rf2 is rf1, f"{mission} failed test of rf2 is rf1")


    #
    # Tests default_bandpass -> (u.nm, u.nm)
    #
    def test_default_bandpass(self):
        """ Tests default_bandpass property returns correct bandpass. """
        bandpass = Tess.get_default_bandpass()
        self.assertEqual(bandpass, (600 * u.nm, 1000 * u.nm))

        bandpass = Kepler.get_default_bandpass()
        self.assertEqual(bandpass, (420 * u.nm, 900 * u.nm))


    #
    # Tests expected_brightness_ratio(t_eff_a, t_eff_b, bandpass)
    #
    def test_expected_brightness_ratio_invalid_args(self):
        """ Tests that expected_brightness_ratio(invalid args) raises errors """        
        for mission in self._all_missions:

            self.assertRaises(TypeError, mission.expected_brightness_ratio,
                              None, 6000 * u.K)
            self.assertRaises(TypeError, mission.expected_brightness_ratio,
                              5000 * u.K, None)

            def_bandpass = mission.get_default_bandpass()
            bad_bandpass = (600 * u.K, 1000 * u.m)

            self.assertRaises(u.UnitsError, mission.expected_brightness_ratio,
                              5000 * u.nm, 6000 * u.K, def_bandpass)
            self.assertRaises(u.UnitsError, mission.expected_brightness_ratio,
                              5000 * u.K, 6000 * u.nm, def_bandpass)
            self.assertRaises(u.UnitsError, mission.expected_brightness_ratio,
                              5000 * u.K, 6000 * u.K, bad_bandpass)

    def test_expected_brightness_ratio_valid_tess_cw_eri(self):
        """ Tests that expected_brightness_ratio(CW Eri/TESS) gives an appropriate result """
        for bandpass in (None, Tess.get_default_bandpass()):
            t_eff_a = 6839 * u.K
            t_eff_b = 6561 * u.K
            ratio = Tess.expected_brightness_ratio(t_eff_a, t_eff_b, bandpass)
            self.assertAlmostEqual(ratio, 0.9, 1)

    def test_expected_brightness_ratio_valid_tess_v1022_cas(self):
        """ Tests that expected_brightness_ratio(V1022 Cas/TESS) gives an appropriate result """
        for bandpass in (None, Tess.get_default_bandpass()):
            t_eff_a = 6450 * u.K
            t_eff_b = 6590 * u.K
            ratio = Tess.expected_brightness_ratio(t_eff_a, t_eff_b, bandpass)
            self.assertAlmostEqual(ratio, 1.1, 1)

if __name__ == "__main__":
    unittest.main()
