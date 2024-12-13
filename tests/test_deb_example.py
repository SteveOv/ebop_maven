""" Tests for the deb_example module. """
import unittest

from ebop_maven.deb_example import create_mags_key

class Test_deb_example(unittest.TestCase):   
    """ Tests for the deb_example module """
    # pylint: disable=invalid-name

    #
    #   TEST create_mags_key(mags_bins: int) -> str
    #
    def test_create_mags_key_integers(self):
        """ Tests create_map_key() make sure that int values are formatted as mags_int"""
        self.assertEqual("mags_1024", create_mags_key(1024))
        self.assertEqual("mags_512", create_mags_key(512.0))


if __name__ == "__main__":
    unittest.main()
