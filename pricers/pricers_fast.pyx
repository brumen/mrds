cimport numpy as np

# declarations of external functions
cdef extern from "math.h":
    double sqrt(double)

cdef extern from "math.h":
    double log(double)

cdef extern from "math.h":
    double exp(double)

cdef extern from "math.h":
    double M_PI

cdef double pdf(double x):
    return exp(-x**2 / 2.0) / sqrt(2 * M_PI)  # SQRT(2 *PI) CAN BE OBTAINED FROM math.h

cpdef double cdf(double x):
    cdef double L, K, w

    # optimized L = abs (x)
    L = -x if x < 0 else x
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / sqrt(2 * M_PI) * exp(-L *L / 2) * (0.31938153 * K - 0.356563782 * K*K + 1.781477937 * K**3 -1.821255978 * K**4 + 1.330274429 * K**5)

    if x < 0 :
        return 1. - w

    return w


## same as black_greeks in pricers module
def black_greeks_fast(double S_0, double K, double r, double sigma, double T, int call_put_ind):
    cdef double d1 = ( log(S_0/K) + 0.5 *sigma**2 * T ) / (sigma * sqrt (T) )
    cdef double d2 = d1 - sigma * sqrt (T)

    cdef double N1 = cdf( d1 )
    cdef double N2 = cdf( d2 )
    cdef double n1 = pdf( d1 )
    cdef double disc = exp ( - r * T)
    cdef double sqrtT = sqrt (T)

    cdef double black = disc * ( S_0 * N1 - K * N2 )
    cdef double delta = disc * N1
    cdef double gamma = disc * n1 / (S_0 * sigma * sqrtT )
    cdef double vega =  disc * S_0 * n1 * sqrtT
    cdef double theta = -S_0 * disc * n1 * sigma / ( 2 * sqrtT) + \
            r * S_0 * disc * N1 - r * K * disc * N2
    cdef double rho = K * T * disc * N2

    if call_put_ind == 1:
        black += (K - S_0) * disc
        delta = disc * ( N1 -1 )
        gamma = disc * n1 / ( S_0 * sigma * sqrtT )
        vega =  disc *  S_0 * n1 * sqrtT
        theta = -S_0 * n1 * sigma / ( 2 * sqrtT ) + \
                r * K * disc * cdf(-d2) - r * S_0 * disc * cdf(-d1)
        rho = - K * T * disc * cdf(-d2)

    return black, delta, gamma, vega, theta, rho


# super fast black call option
cpdef double black_call_fast(double S_0, double K, double r, double sigma, double T):

    cdef double d1 = (log(S_0/K) + 0.5 *sigma**2 * T) / (sigma * sqrt (T))
    cdef double d2 = d1 - sigma * sqrt(T)

    return exp(-r * T) * (S_0 * cdf(d1) - K * cdf(d2))

# super fast black put option
cpdef double black_put_fast (double S_0, double K, double r, double sigma, double T):

    cdef double d1 = ( log(S_0/K) + 0.5 *sigma**2 * T ) / (sigma * sqrt (T) )
    cdef double d2 = d1 - sigma * sqrt (T)

    return exp(-r * T) * ( K * cdf(-d2) - S_0 * cdf (-d1) )


# fast version of the pricers.py version
cdef double spread_option_kirk_fast (double F_1, double F_2, double K,
                                     double sigma_1, double sigma_2, double rho,
                                     double T, double DF):

    cdef double sigma_K = sqrt ( sigma_1**2 - 2 * F_2 / (F_2 + K) * rho * sigma_1 * sigma_2 + (F_2 / (F_2 + K)) ** 2 * sigma_2**2 )
    cdef double d_1 = ( log (F_1 / (F_2+K) ) + 0.5 * sigma_K**2 * T ) / (sigma_K * sqrt (T) )

    return  DF * (F_1 * cdf(d_1) - (F_2+K)*cdf(d_1 - sigma_K * sqrt (T)))
