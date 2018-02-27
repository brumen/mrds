from pricers_fast cimport pdf
from pricers_fast cimport cdf 

cdef extern from "math.h":
    double sqrt (double)

cdef extern from "math.h":
    double log (double)

cdef extern from "math.h":
    double exp (double)


cpdef double transition_mtx_ln_blocks_fast(double p_dash, double p, double op,
                                           double F_P, double F_OP,
                                           double sigma_P, double sigma_OP,
                                           double rho, double t, double delta_t):


    d1 = (log(p_dash/p) + 0.5 * sigma_P**2 * delta_t) / (sigma_P * sqrt(delta_t) )
    d2 = (log(p/F_P) + 0.5 * sigma_P**2 * t) / (sigma_P * sqrt(t) )
    z_p = d2
    z_op = (log(op/F_OP) + 0.5 * sigma_OP**2 * t) / (sigma_OP * sqrt(t) )
    d3 = (z_p - rho * z_op) / sqrt(1. - rho**2)

    return cdf(d1) * pdf(d3) * pdf(d2)


#
# internal function to integrate only, used for sparse grids 
#  v... integrating variable replaces p
cpdef double transition_mtx_ln_blocks_fast_internal (double p_dash, double v, double op,
                                                     double F_P, double F_OP,
                                                     double sigma_P, double sigma_OP,
                                                     double rho, double t, double delta_t):

    z_op = (log(op/F_OP) + 0.5 * sigma_OP**2 * t) / (sigma_OP * sqrt(t) )
    p = F_P * exp( (sqrt(1.-rho**2) * v + rho * z_op)*sigma_P * sqrt(t) -
                          0.5 * sigma_P**2 * t )
    z_p = (log(p/F_P) + 0.5 * sigma_P**2 * t) / (sigma_P * sqrt(t) )
    v = (z_p - rho * z_op) / sqrt(1. - rho**2)
    d1 = (log(p_dash/p) + 0.5 * sigma_P**2 * delta_t) / (sigma_P * sqrt(delta_t) )

    return cdf(d1) * sigma_P * sqrt(t) * p # n(v) is omitted, as Guass-Hermite handles it 

