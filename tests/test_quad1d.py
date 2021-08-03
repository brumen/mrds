""" Tests for sparse grid integration.
"""

import numpy as np

from unittest import TestCase

from mrds.quad1d import gh_quad, gh_quad_nd
from scipy.integrate import quad, dblquad


class Quad1DTests(TestCase):

    def test_1d_gh(self):
        """ Test 1 dimensional gauss hermite integration
        """

        res_1 = gh_quad(lambda x: x**2, 10)
        res_2 = quad(lambda x: x**2 * np.exp(-x**2/2) / np.sqrt(np.pi * 2) , a = -np.inf, b=np.inf)[0]

        self.assertAlmostEqual(res_1, res_2, 3)

    def test_nd_gh(self):
        """ N-dimensional gauss quadrature.
        """

        res_1 = gh_quad_nd(lambda x, y: x**2 + y**2, (10, 10))
        res_2 = dblquad( lambda x, y: (x**2 + y**2) * np.exp(-x**2/2 - y**2/2) / (np.pi * 2)
                       , a = -np.inf
                       , b = np.inf
                       , gfun = - np.inf
                       , hfun = np.inf )[0]

        self.assertAlmostEqual(res_1, res_2, 3)
