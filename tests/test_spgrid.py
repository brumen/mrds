""" Tests for sparse grid integration.
"""

import numpy as np
import scipy.integrate

from unittest import TestCase

from mrds.sg import sg_quad


class SparseGridTests(TestCase):

    def test_simple_1(self):
        """ Test the sparse grid quadrature in 1 dim.
        """

        res_1 = sg_quad (1, 15, lambda x: x**2)
        res_2 = scipy.integrate.quad (lambda x: x**2 * np.exp(-x**2/2)/np.sqrt(2. * np.pi), -np.inf, np.inf)[0]

        self.assertAlmostEqual( res_1, res_2, 2 )

    def test_sg_quad(self):
        """ Integrate (2 * x[0]**2 + 3 * x[1]**3) * e^(-x**2 - y**2) / 2 pi
        """

        g = lambda x: 2 * x[0]**2 + 3 * x[1]**3
        v_1 = sg_quad(2, 5, g)
        v_2 = scipy.integrate.dblquad( lambda x, y: g([x, y]) * np.exp(-x**2/2. - y**2/2.)
                                    , -np.inf, np.inf
                                    , -np.inf, np.inf )[0] / (2 * np.pi)

        self.assertAlmostEqual(v_1, v_2, 5)
