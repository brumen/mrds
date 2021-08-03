""" 1d, and n-dim sparse grid integration.
"""

import numpy as np

from numpy         import sqrt, pi
from typing        import Callable, Tuple
from scipy.special import roots_hermite


def gh_quad(f : Callable, n : int) -> float:
    """ 1 dimensional hermite quadrature 1/sqrt(2pi) * int_R e^{-x^2/2} f(x) dx

    :param n: order of integration
    :param f: function of 1 variable to integrate
    :returns the integration as described above.
    """

    points, weights = roots_hermite(n)

    return np.sum([ w * f(x * sqrt(2.)) for w, x in zip(weights, points) ]) / sqrt(pi)


def gh_quad_nd(f : Callable, nn : Tuple ) -> float:
    """ Integrates a function f of n variables with respect to certain sparse grid in every dimension.
    gauss-hermite in multiple dimensions

    :param f: function to integrate
    :param nn: vector of sparse grid size in each dimension.
    :returns: result of sparse grid integration.
    """

    n_dim = len(nn)

    if n_dim == 1:
        return gh_quad(f, nn[0])  # some version of this

    # we are in higher dimension
    def new_fct(*prev_vars):
        return gh_quad(lambda curr_var: f(*prev_vars, curr_var), nn[-1])  # function

    return gh_quad_nd( new_fct , nn[:-1] )


def main():
    # examples
    gh_quad_nd(lambda x, y, z: np.exp(-x ** 2 - y ** 2 - z ** 2), (10, 10, 10))


# main()
