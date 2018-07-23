#
# Hyperbolic correlations on the vol curve.
#

from numpy import cosh, sqrt, array


def corr_hyp_sec_basic(alpha, i, j):
    """
    Correlation between months i, j.

    """

    return 1.0 / cosh(alpha * (i - j))


def corr_hyp_sec_two_fronts(rho, i, j):
    return corr_hyp_sec_basic(sqrt(2 * (1 - rho)), i, j)


def corr_hyp_sec_mat(rho, ind_range):
    """
    Generates a correlation matrix from the hyp sec function above.

    """

    return array([[corr_hyp_sec_two_fronts(rho, i, j)
                   for j in ind_range]
                  for i in ind_range])
