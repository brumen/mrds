# test file for sparse grids
import config
import numpy as np
import scipy as sp
import scipy.integrate as spi
import sg


def test_sg_quad():
    """
    integrate (2 * x[0]**2 + 3 * x[1]**3) * e^(-x**2 - y**2) / 2 pi

    timing was done and sparse grids are much faster
    """
    g = lambda x: 2 * x[0]**2 + 3 * x[1]**3
    v1 = sg.sg_quad(2, 5, g)
    v2 = spi.dblquad(lambda x, y: g([x, y]) * np.exp(-x**2/2. - y**2/2.),
                     -np.inf, np.inf,
                     lambda x: -np.inf, lambda x: np.inf)[0] / (2 * np.pi)
    res = np.abs(v1 - v2) < 1.e-5
    return res

