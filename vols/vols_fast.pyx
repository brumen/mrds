# Fast version of the vol functions

cimport numpy as np
np.import_array()

from mrds.pricers.pricers_fast import cdf  # This _HAS_ to be import and _not_ cimport

# declarations of external functions
cdef extern from "math.h":
    double sqrt(double)

cdef extern from "math.h":
    double log(double)

cdef extern from "math.h":
    double exp(double)

cdef extern from "math.h":
    double fabs(double)

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

cdef extern from "numpy/npy_math.h":
    double NPY_PI


# normalized strike of K
cdef double norm_strike(double S0, double K, double sigma, double ttm):
    return log(K/S0) / (sigma * sqrt(ttm))


cdef double rational_approximation (double t):
    # Abramowitz and Stegun formula 26.2.23
    # absolute value of the error should be less than 4.5 e-4
    return t - ((0.010328 * t + 0.802853)*t + 2.515517)/ (((0.001308*t + 0.189269)*t + 1.432788)*t + 1.)


cdef double normal_ppf(double p):
    if p < 0.5:
        return - rational_approximation(sqrt(-2.*log(p)))

    return rational_approximation(sqrt(-2.*log(1.-p)))


cdef double b(double x, double sigma, double theta):
    cdef double d1
    cdef double e1 = exp (x/2)

    if sigma != 0.:
        d1 = x/sigma + sigma /2.
        return theta * e1 * cdf(theta * d1) - theta/e1 * cdf(theta*(d1 - sigma))

    if (theta >= 0 and x >= 0) or (theta <= 0 and x <= 0):  # d1 = +infty or d1 = -infty
        return theta * e1 - theta / e1

    return 0.


cdef double sigma_c(double x):
    return sqrt(2. * fabs(x))  # inflection point

cdef double iota(double x, double theta):
    return (theta *x > 0.)*theta*(exp(x/2.) - exp(-x/2.))


cdef b_c(x, theta):
    return b(x, sigma_c(x), theta)


cdef double one_step(x, sigma, theta, beta):
    cdef double b_der
    b_der = exp(-0.5 * (x/sigma)**2 - 0.5 * (sigma/2)**2) / sqrt(2. * 3.1415)

    if (beta < b_c(x, theta) ):
        return log((beta - iota(x, theta)) / (b(x, sigma, theta) - iota(x, theta))) * \
            (b(x, sigma, theta) - iota(x, theta)) / b_der

    return (beta - b(x,sigma, theta)) / b_der


def black_vol_inverse_normalized(double beta, double x, double theta, double tol):
    """ Solving for sigma: b_fct (x, sigma, theta ) = beta

    @param tol: tolerance level
    @returns: sigma * sqrt (t)
    """

    cdef double sigma, sigma_new, delta_sigma, e1

    if (beta < b_c(x,theta)):
        sigma = sqrt((2. * x**2 ) / (fabs (x) - 4. * log ((beta - iota (x,theta))/ \
                                                                 (b_c(x,theta) - iota(x, theta)))))
    else:
        e1 = exp(theta * x / 2.0)
        sigma = - 2. * normal_ppf((e1 - beta) / (e1 - b_c(x,theta)) * cdf (-sqrt(fabs(x)/2.0)))

    sigma_new   = sigma * (1. + 2. * tol )
    delta_sigma = 2. * tol

    # TODO: ERROR HANDLING HAS TO BE HERE

    if sigma <= 0.:
        sigma = 1.e-16

    while (delta_sigma / sigma > tol ):
        sigma_new   = sigma + one_step(x, sigma, theta, beta)
        delta_sigma = fabs(sigma_new - sigma)
        sigma       = sigma_new

    if sigma_new <= 0.:
        return tol

    return sigma_new
