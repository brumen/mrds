#
# Mathematical functions for the Commodity skew model.
#

import numpy as np
import scipy
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats
import scipy.optimize
import scipy.interpolate


class ComMathsMixin:
    """ Mathematical functions of the Commodity skew.
    """

    # where to cut off the distribution function.
    _SMALL_CUTOFF = -1e10
    _LARGE_CUTOFF = 1e10

    @staticmethod
    def _trunc_normal_above(a: float) -> np.array:
        """ Computes the truncated E[ N^{0,1,2,3,4} * 1(N <a) ] where N std. normal in succession.

        :param a: parameter for the truncation.
        :returns: distribution of the truncated std. normal evaluated at a.
        """

        if a < ComMathsMixin._SMALL_CUTOFF:
            return np.array([0., 0., 0., 0., 0.])

        if a > ComMathsMixin._LARGE_CUTOFF:
            return np.array([1., 0., 1., 0., 3.])

        # most common case
        sqrt_2 = np.sqrt(2.)
        sqrt_2pi = np.sqrt(2. * np.pi)

        return np.array([scipy.stats.norm.cdf(a)
                            , - np.exp(- a ** 2 / 2.) / sqrt_2pi
                            , 0.5 + 0.5 * scipy.special.erf(a / sqrt_2) - np.exp(-a ** 2 / 2.) * a / sqrt_2pi
                            , - (a ** 2 + 2.) * np.exp(- a ** 2 / 2.) / sqrt_2pi
                            , - a * (a ** 2 + 3.) * np.exp(- a ** 2 / 2.) / sqrt_2pi + 1.5 * (
                                     1. + scipy.special.erf(a / sqrt_2))])

    @staticmethod
    def _trunc_normal_below(a: float) -> np.array:
        """
        computes the truncated E[ N^{0,1,2,3,4} * 1(N >a) ] where N std. normal.

        :param a: parameter for the truncation.
        :returns truncated std. normal.
        """

        return - ComMathsMixin._trunc_normal_above(a) + np.array([1., 0., 1., 0., 3.])

    @staticmethod
    def _trunc_normal_interval(a: float, b: float) -> np.array:
        """
        Truncated standard normal for the interval [a,b]

        :param a: beginning of the interval
        :param b: end of the interval
        """

        return ComMathsMixin._trunc_normal_above(b) - ComMathsMixin._trunc_normal_above(a)

    @staticmethod
    def __simulate_std_normal( nb_factors     : int
                             , corr_mtx       : np.array
                             , nb_simulations : int ) -> np.ndarray:
        """ Simulates the standard normal random variables with specified correlations, and returns the
        matrix of size (nb_factors x nb_correlations)  # TODO: CHECK IF THIS IS NOT REVERSED.

        :param corr_mtx: correlation matrix, a nb_factors x nb_factors matrix.
        :param nb_simulations: number of simulations from the factors.
        """

        return np.random.multivariate_normal( np.zeros(nb_factors)
                                            , corr_mtx
                                            , size = nb_simulations )
