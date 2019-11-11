#
# Hyperbolic correlations on the vol curve.
#

import datetime

from numpy import cosh, sqrt, array


def corr_hyp_sec_basic(alpha : float, i : int, j : int ):
    """ Correlation between months i, j.

    :param alpha: correlation parameter
    :param i: first month integer
    :param j: second month integer.
    """

    return 1.0 / cosh(alpha * (i - j))


def corr_hyp_sec_time_diff(alpha : float, date_1 : datetime.date, date_2 : datetime.date, dcf = 365.25):
    """ Similar to the function above, but with more flexibility

    :param alpha:
    """

    return 1.0 / cosh(alpha * (date_2 - date_1).days/dcf)


def corr_hyp_sec_two_fronts(rho : float, i : int, j : int) -> float:
    return corr_hyp_sec_basic(sqrt(2 * (1 - rho)), i, j)


def corr_hyp_sec_two_fronts_time_diff(rho : float, date_1 : datetime.date, date_2 : datetime.date) -> float:
    return corr_hyp_sec_time_diff(sqrt(2 * (1 - rho)), date_1, date_2)


def corr_hyp_sec_mat(rho, ind_range):
    """
    Generates a correlation matrix from the hyp sec function above.

    """

    return array([[corr_hyp_sec_two_fronts(rho, i, j)
                   for j in ind_range]
                  for i in ind_range])
