import scipy
import scipy.integrate
import scipy.linalg
import unittest
import numpy as np

# testing sparse grids
import mrds.sg as sg

import scipy.integrate


class SparseGridTests(unittest.TestCase):

    def test_simple_1(self):
        A = sg.sg_p (1,5)
        B = sg.sg_w (1,5)
        f = lambda x: sum (x**2)
        res1 = sg.sg_quad (1, 15, f)
        res2 = scipy.integrate.quad (f, 0., 1.)[0]
        res3 = scipy.linalg.norm(res1-res2) < 1.

        # really WEAK test
        self.assertTrue( res3 )

    def test_sg_quad(self):
        """ Integrate (2 * x[0]**2 + 3 * x[1]**3) * e^(-x**2 - y**2) / 2 pi
        """

        g = lambda x: 2 * x[0]**2 + 3 * x[1]**3
        v1 = sg.sg_quad(2, 5, g)
        v2 = scipy.integrate.dblquad(lambda x, y: g([x, y]) * np.exp(-x**2/2. - y**2/2.),
                         -np.inf, np.inf,
                         lambda x: -np.inf, lambda x: np.inf)[0] / (2 * np.pi)

        self.assertLess(np.abs(v1 - v2), 1.e-5)


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(SparseGridTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
