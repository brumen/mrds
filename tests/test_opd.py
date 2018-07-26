from opd import opd_avx
import numpy as np
from unittest import TestCase


class TestOpd(TestCase):

    def setUp(self):

        self.N = 128000
        self.r = np.random.rand(self.N)
        self.a = np.random.rand(self.N)
        self.b = np.random.rand(self.N)
        self.c = np.random.rand(self.N)
        self.d = np.random.rand(self.N)
        self.y = np.empty(self.N)

    def test_just_run(self):
        """
        Tests whether add4 function works correctly.

        """

        y1 = np.empty(self.N)
        opd_avx.add4(self.r, self.a, self.b, self.c, self.d, y1, self.N)
        y2 = self.r - (self.a + self.b + self.c + self.d)

        self.assertTrue(y1 == y2)
