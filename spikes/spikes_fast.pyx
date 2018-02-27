# Fast version of the local-vol functions 

cimport numpy as np

import pricers_fast

# declarations of external functions 
cdef extern from "math.h":
    double sqrt (double)

cdef extern from "math.h":
    double log (double)

cdef extern from "math.h":
    double exp (double)


# 
# normalized strike of K
cpdef int test1 (int n):
    cdef int k
    cdef int su = 0
    for k in range (n):
        su = su + k 
	 
    return su


cpdef europe_price_spike (double S_0, double K, double sigma, double T, 
                          double eta, double lam, double delta, double disc_fact):
        d_1 = log ( S_0 / K * (1.+ eta) ) + 0.5 * sigma**2 * T
        #disc_fact = self.DF(T)
        r = - log (disc_fact) / T
        return (1- lam * delta) * pricers_fast.black_call_fast (S_0, K, r, sigma, T) \
            + lam * delta * pricers_fast.black_call_fast (S_0, K/(1. + eta ), r, sigma, T) \
            + disc_fact * lam * delta * eta * S_0 * pricers_fast.cdf ( d_1 ) 
