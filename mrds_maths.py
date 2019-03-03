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
import scipy.interpolate  # spline package


class ComMaths:
    """
    Mathematical functions of the Commodity skew.

    """

    @staticmethod
    def __trunc_normal_above__(a: float) -> np.array:
        """
        Computes the truncated E[ N^{0,1,2,3,4} * 1(N <a) ] where N std. normal in succession.

        :param a: parameter for the truncation.
        :returns: truncated std. normal variable.
        """

        if a < -1e10:
            return np.array([0., 0., 0., 0., 0.])

        if a > 1e10:
            return np.array([1., 0., 1., 0., 3.])

        # most common case
        sqrt_2 = np.sqrt(2.)
        sqrt_2pi = np.sqrt(2. * np.pi)

        return np.array([scipy.stats.norm.cdf(a)
                            , - np.exp(- a ** 2 / 2.0) / sqrt_2pi
                            , 0.5 + 0.5 * scipy.special.erf(a / sqrt_2) - np.exp(-a ** 2 / 2.) * a / sqrt_2pi
                            , - (a ** 2 + 2.) * np.exp(- a ** 2 / 2.) / sqrt_2pi
                            , - a * (a ** 2 + 3.) * np.exp(- a ** 2 / 2.) / sqrt_2pi + 1.5 * (
                                     1. + scipy.special.erf(a / sqrt_2))])

    @staticmethod
    def __trunc_normal_below__(a: float) -> np.array:
        """
        computes the truncated E[ N^{0,1,2,3,4} * 1(N >a) ] where N std. normal.

        :param a: parameter for the truncation.
        :returns truncated std. normal.
        """

        return - ComMaths.__trunc_normal_above__(a) + np.array([1.0, 0.0, 1.0, 0.0, 3.0])

    @staticmethod
    def __trunc_normal_interval__(a: float, b: float) -> np.array:
        """
        Truncated standard normal for the interval [a,b]

        :param a: beginning of the interval
        :param b: end of the interval
        """

        return ComMaths.__trunc_normal_above__(b) - ComMaths.__trunc_normal_above__(a)
