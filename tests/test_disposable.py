""" Unit tests for disposable module. """
import unittest
from ebop_maven.disposable import do_something

class Testdisposable(unittest.TestCase):
    """ Unit tests for disposable module. """

    def test_do_something(self):
        """ Tests do_something(). """
        val = do_something()
        self.assertEqual(val, 2112)

if __name__ == "__main__":
    unittest.main()
