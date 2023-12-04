import unittest

from snuqs.circuit import *
import random
import math


class QopTest(unittest.TestCase):
    def test_non_list_qreg(self):
        q = Qreg("q", 5)
        self.assertRaises(TypeError, lambda: ID(Qreg("q", 5)))

if __name__ == '__main__':
    unittest.main()
