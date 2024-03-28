""" Unit tests for the StellarModels base class and sub classes. """
import unittest
from pathlib import Path
from inspect import getsourcefile
from argparse import Namespace

import astropy.units as u
from astropy.units.core import UnitsError

from ebop_maven.libs.stellarmodels import StellarModels, MistStellarModels, ParsecStellarModels

# pylint: disable=too-many-public-methods, line-too-long
class TestStellarModels(unittest.TestCase):
    """ Unit tests for the StellarModels base class and sub classes. """
    __this_dir = Path(getsourcefile(lambda:0)).parent

    # List of the StellarModels subclasses
    _known_models = [MistStellarModels, ParsecStellarModels]

    # The known model short names and a known combinations of matching stellar params
    _test_data = {
        "Mist": {
            "Sol": Namespace(**{
                "Z": 0.0143, 
                "Y": 0.2703,
                "INIT_MASS": 1.0 * u.solMass,
                "AGE": 9.7 * u.dex(u.yr),
                "M": 1.0 * u.solMass,
                "R": 1.0 * u.solRad,
                "L": 1.0 * u.solLum,
                "TE": 5850 * u.K,
                "LOGG": 4.4 * u.dex(u.cm / u.s**2),
                "PHASE": 0
            })
        },
        "Parsec": {
            "Sol": Namespace(**{
                "Z": 0.014,
                "Y": 0.273,
                "INIT_MASS": 1.0 * u.solMass,
                "AGE": 9.7 * u.dex(u.yr),
                "M": 1.0 * u.solMass,
                "R": 1.0 * u.solRad,
                "L": 1.0 * u.solLum,
                "TE": 5900 * u.K,
                "LOGG": 4.4 * u.dex(u.cm / u.s**2),
                "PHASE": 5
            })
        }
    }

    #
    # Tests StellarModels.get_instance(instance_name: str) -> StellarModels:
    #
    def test_base_get_instance_unknown_models(self):
        """ Tests the base class's cls.get_instance() function with unknown models. """
        self.assertRaises(KeyError, StellarModels.get_instance, "Camalot")
        self.assertRaises(KeyError, StellarModels.get_instance, "It'sOnlyAModel")

    def test_base_get_instance_known_models(self):
        """ Tests the base class's cls.get_instance() function with known models. """
        for name, models_type in [("Mist", MistStellarModels),
                                  ("MistStellarModels", MistStellarModels),
                                  ("PARSEC", ParsecStellarModels),
                                  ("ParsecStellARModels", ParsecStellarModels)]:
            stellar_models = StellarModels.get_instance(name)
            self.assertIsInstance(stellar_models, models_type)

    def test_base_get_instance_assert_caching(self):
        """ Tests the base class's cls.get_instance() to assert caching of instances """
        # Currently uses a simple lru_cache decorator
        # so it's dependent on consistent naming
        instance1 = StellarModels.get_instance("Mist")
        instance2 = StellarModels.get_instance("Mist")
        self.assertEqual(instance1, instance2)


    #
    # Tests subclasses's __init__(data_file)
    #
    def test_init_data_file_not_given(self):
        """ Test __init__() when no data_file specified. """
        for models in self._known_models:
            self.assertIsNotNone(models().row_count)

    def test_init_data_file_exists(self):
        """ Test __init__() when a valid, existing data_file specified. """
        for models in self._known_models:
            self.assertIsNotNone(models(models.default_data_file).row_count)

    def test_init_data_file_not_exist(self):
        """ Test __init__() when an unknown data_file specified raises FileNotFoundError. """
        unknown_data_file = self.__this_dir / "../unknown.pkl.xz"
        for models in self._known_models:
            self.assertRaises(FileNotFoundError, models, unknown_data_file)


    #
    # Tests lookup_stellar_parameters(initial_mass, age, z, y)
    # Now a convenience wrapper on lookup_parameters() so leave validation tests to that
    #
    def test_lookup_stellar_parameters_with_valid_fiducial_request(self):
        """ Tests lookup_stellar_parameters(valid args with fiducial initial_mass) """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                mass, radius, t_eff, lum, logg = inst.lookup_stellar_parameters(params.Z, params.Y, params.INIT_MASS, params.AGE)
                self.assertAlmostEqual(mass.value, params.M.value, 1, msg)
                self.assertAlmostEqual(radius.value, params.R.value, 1, msg)
                self.assertAlmostEqual(t_eff.value, params.TE.value, -3, msg)
                self.assertAlmostEqual(lum.value, params.L.value, 0, msg)
                self.assertAlmostEqual(logg.value, params.LOGG.value, 1, msg)

    def test_lookup_stellar_parameters_with_valid_different_units(self):
        """ Tests lookup_stellar_parameters(valid args with SI initial_mass) """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                si_init_mass = params.INIT_MASS.to(u.kg)
                linear_age = params.AGE.to(u.Gyr)
                mass, radius, t_eff, lum, logg = inst.lookup_stellar_parameters(params.Z, params.Y, si_init_mass, linear_age)
                self.assertAlmostEqual(mass.value, params.M.value, 1, msg)
                self.assertAlmostEqual(radius.value, params.R.value, 1, msg)
                self.assertAlmostEqual(t_eff.value, params.TE.value, -3, msg)
                self.assertAlmostEqual(lum.value, params.L.value, 0, msg)
                self.assertAlmostEqual(logg.value, params.LOGG.value, 1, msg)

    #
    # Tests lookup_parameters(initial_mass, age, z, y, parameter_names)
    #
    def test_lookup_parameters_args_units_validation(self):
        """ Tests lookup_parameters(various invalid index units or types) raises Type/UnitError as appropriate. """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for _, params in stars.items():
                self.assertRaises(TypeError,  inst.lookup_parameters, None, params.Y, params.INIT_MASS, params.AGE)
                self.assertRaises(TypeError,  inst.lookup_parameters, params.Z,  None, params.INIT_MASS, params.AGE)
                self.assertRaises(TypeError,  inst.lookup_parameters, params.Z, params.Y, None, params.AGE)
                self.assertRaises(TypeError,  inst.lookup_parameters, params.Z, params.Y, 1.2, params.AGE)
                self.assertRaises(UnitsError, inst.lookup_parameters, params.Z, params.Y, 1.2 * u.solRad, params.AGE)
                self.assertRaises(TypeError,  inst.lookup_parameters, params.Z, params.Y, params.INIT_MASS, None)
                self.assertRaises(TypeError,  inst.lookup_parameters, params.Z, params.Y, params.INIT_MASS, 9)
                self.assertRaises(UnitsError, inst.lookup_parameters, params.Z, params.Y, params.INIT_MASS, 9 * u.K)

    def test_lookup_parameters_with_unknown_key_value(self):
        """ Tests lookup_parameters(various unknown key values) raises KeyError. """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for _, params in stars.items():
                self.assertRaises(KeyError, inst.lookup_parameters, 0, params.Y, params.INIT_MASS, params.AGE)
                self.assertRaises(KeyError, inst.lookup_parameters, params.Z,  0, params.INIT_MASS, params.AGE)
                self.assertRaises(KeyError, inst.lookup_parameters, params.Z, params.Y, 0 * u.solMass, params.AGE)
                self.assertRaises(KeyError, inst.lookup_parameters, params.Z, params.Y, params.INIT_MASS, 0 * u.dex(u.yr))

    def test_lookup_parameters_with_unknown_parameter_name(self):
        """ Tests lookup_parameters(various invalid/unkown parameter names) raises ValueError/TypeError. """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for _, params in stars.items():
                self.assertRaises(ValueError, inst.lookup_parameters, params.Z, params.Y, params.INIT_MASS, params.AGE, ["R", "Unknown"])
                self.assertRaises(TypeError, inst.lookup_parameters, params.Z, params.Y, params.INIT_MASS, params.AGE, ["R", None])

    def test_lookup_parameters_with_known_parameter_names(self):
        """ Tests lookup_parameters(various kown parameter names) returns requested data in order required. """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                for param_names, exp_results, assert_places in [
                    (["M"], [params.M], [1]),
                    (["TE", "M", "LOGG"], [params.TE, params.M, params.LOGG], [-2, 1, 1])
                ]:
                    results = inst.lookup_parameters(params.Z, params.Y, params.INIT_MASS, params.AGE, param_names)
                    for ix, (res, exp, places) in enumerate(zip(results, exp_results, assert_places)):
                        msg = f"{models_name}/{star_name}/{param_names}/{ix}"
                        self.assertAlmostEqual(res.value, exp.value, places, msg)


    #
    # Tests list_distinct_z_values(y_filter=None)
    #
    def test_list_distinct_z_values_with_y_filter_none(self):
        """ Test list_distinct_z_values(y_filter is None). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_z_values()
                self.assertTrue(len(values) > 0, msg)
                self.assertIn(params.Z, values, msg)

    def test_list_distinct_z_values_with_y_filter_known(self):
        """ Test list_distinct_z_values(y_filter is known). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_z_values(params.Y)
                self.assertTrue(len(values) > 0, msg)
                self.assertIn(params.Z, values, msg)

    def test_list_distinct_z_values_with_y_filter_unknown(self):
        """ Test list_distinct_z_values(y_filter is unknown). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name in stars:
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_z_values(0.99)
                self.assertTrue(len(values) == 0, msg)


    #
    # Tests list_distinct_y_values(z_filter)
    #
    def test_list_distinct_y_values_with_z_filter_none(self):
        """ Test list_distinct_y_values(z_filter is None). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_y_values()
                self.assertTrue(len(values) > 0, msg)
                self.assertIn(params.Y, values, msg)

    def test_list_distinct_y_values_with_z_filter_known(self):
        """ Test list_distinct_z_values(z_filter is known). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_y_values(params.Z)
                self.assertTrue(len(values) > 0, msg)
                self.assertIn(params.Y, values, msg)

    def test_list_distinct_y_values_with_z_filter_unknown(self):
        """ Test list_distinct_z_values(z_filter is unknown). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name in stars:
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_y_values(0.99)
                self.assertTrue(len(values) == 0, msg)


    #
    # Tests list_distinct_initial_masses()
    #
    def test_list_distinct_initial_masses(self):
        """ Test list_distinct_initial_masses with no filter. """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_initial_masses()
                self.assertTrue(len(values) > 0, msg)
                self.assertIn(params.INIT_MASS, values, msg)


    #
    # Tests list_distinct_ages(initial_mass_filter)
    #
    def test_list_distinct_ages_filter_validation(self):
        """ Test list_distinct_ages() raises unit validation errors on initial_mass_filter """
        for models_name in self._test_data:
            inst = StellarModels.get_instance(models_name)
            self.assertRaises(UnitsError, inst.list_distinct_ages, 1.0 * u.solRad)
            self.assertRaises(TypeError, inst.list_distinct_ages, 1.0)

    def test_list_distinct_ages_with_no_filters(self):
        """ Test list_distinct_initial_ages(initial_mass_filter is None). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_ages()
                self.assertTrue(len(values) > 0, msg)
                self.assertIn(params.AGE, values, msg)

    def test_list_distinct_ages_with_initial_mass_filter_known(self):
        """ Test list_distinct_initial_ages(initial_mass_filter is known). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name, params in stars.items():
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_ages(initial_mass_filter=params.INIT_MASS)
                self.assertTrue(len(values) > 0, msg)
                self.assertIn(params.AGE, values, msg)

    def test_list_distinct_ages_with_initial_mass_filter_unknown(self):
        """ Test list_distinct_initial_ages(initial_mass_filter is unknown). """
        for models_name, stars in self._test_data.items():
            inst = StellarModels.get_instance(models_name)
            for star_name in stars:
                msg = f"{models_name}/{star_name}"
                values = inst.list_distinct_ages(initial_mass_filter = 5.0512 * u.solMass)
                self.assertTrue(len(values) == 0, msg)


if __name__ == "__main__":
    unittest.main()
