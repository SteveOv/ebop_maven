""" Unit tests for disposable module. """
import unittest
from ebop_maven.disposable import do_something
from ebop_maven.libs.functions import add

class Testdisposable(unittest.TestCase):
    """ Unit tests for disposable module. """

    def test_do_something(self):
        """ Tests do_something(). """
        val = do_something()
        self.assertEqual(val, 2112)

    def test_libs_functions_add(self):
        """ Tests add() from libs subpackage. """
        self.assertEqual(2, add(1, 1))

if __name__ == "__main__":
    unittest.main()
