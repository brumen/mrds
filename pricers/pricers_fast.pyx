# TODO: THIS LINE HERE IS WEAK, ONLY IMPORT FILES THAT 
cimport numpy as np

# declarations of external functions 
cdef extern from "math.h":
    double sqrt(double)

cdef extern from "math.h":
    double log(double)

cdef extern from "math.h":
    double exp(double)

# EXTERNAL LIBRARY DEPENDENCY USE 
# perhaps these two fcs. can also be obtained from the 
cdef double pdf(double x):
    return exp(-x**2 / 2.0) / sqrt(2 * np.pi)

cpdef double cdf(double x):
    cdef double L, K, w

    # optimized L = abs (x) 
    if x < 0:
        L = -x
    else:
        L = x

    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / sqrt(2 * np.pi) * exp(-L *L / 2) * (0.31938153 * K - 0.356563782 * K*K + 1.781477937 * K**3 -1.821255978 * K**4 + 1.330274429 * K**5)

    if x < 0 :
        return 1. - w

    return w


## same as black_greeks in pricers module
def black_greeks_fast(double S_0, double K, double r, double sigma, double T, int call_put_ind):
    #if abs(sigma) < 1e-10:
    #    return array ([exp ( - r * T ) * (S_0 - K), 1, 0, 0, 0, 0 ])
    #else:
    cdef double d1 = ( log(S_0/K) + 0.5 *sigma**2 * T ) / (sigma * sqrt (T) )
    cdef double d2 = d1 - sigma * sqrt (T)

    cdef double N1 = cdf( d1 )
    cdef double N2 = cdf( d2 )
    cdef double n1 = pdf( d1 )
    cdef double disc = exp ( - r * T)
    cdef double sqrtT = sqrt (T)

    cdef double Black = disc * ( S_0 * N1 - K * N2 )
    cdef double Delta = disc * N1
    cdef double Gamma = disc * n1 / (S_0 * sigma * sqrtT )
    cdef double Vega =  disc * S_0 * n1 * sqrtT
    cdef double Theta = -S_0 * disc * n1 * sigma / ( 2 * sqrtT) + \
            r * S_0 * disc * N1 - r * K * disc * N2
    cdef double Rho = K * T * disc * N2

    if (call_put_ind == 1):
        Black = Black + (K - S_0) * disc
        Delta = disc * ( N1 -1 )
        Gamma = disc * n1 / ( S_0 * sigma * sqrtT )
        Vega =  disc *  S_0 * n1 * sqrtT
        Theta = -S_0 * n1 * sigma / ( 2 * sqrtT ) + \
                r * K * disc * cdf(-d2) - r * S_0 * disc * cdf(-d1)
        Rho = - K * T * disc * cdf(-d2)

    return np.array([Black, Delta, Gamma, Vega, Theta, Rho])

# super fast black call option 
cpdef double black_call_fast(double S_0, double K, double r, double sigma, double T):

    cdef double d1 = (log(S_0/K) + 0.5 *sigma**2 * T) / (sigma * sqrt (T))
    cdef double d2 = d1 - sigma * sqrt(T)

    return exp(-r * T) * (S_0 * cdf(d1) - K * cdf(d2))

# super fast black put option 
cdef double black_put_fast (double S_0, double K, double r, double sigma, double T):

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


# computes the kirk formula for the entire matrices 
cpdef spread_option_kirk_mat (np.ndarray[np.float64_t, ndim=2] F_1_m, 
                              np.ndarray[np.float64_t, ndim=2] F_2_m, double K, 
                              np.ndarray[np.float64_t, ndim=2] impl_surf_1, 
                              np.ndarray[np.float64_t, ndim=2] impl_surf_2, 
                              double rho, 
                              np.ndarray[np.float64_t, ndim=1] T_v, double DF):

    cdef int ttm_ind 
    cdef int K_ind 

    cdef np.ndarray[np.float64_t, ndim=2] res_mat = np.zeros ( (np.shape(F_1_m)[0], len(T_v)) )

    for ttm_ind in range(len(T_v)):
        for K_ind in range (np.shape(F_1_m)[0]):
            res_mat[K_ind,ttm_ind] = spread_option_kirk_fast (F_1_m[K_ind,ttm_ind], 
                                                              F_2_m[K_ind,ttm_ind], 
                                                              K, 
                                                              impl_surf_1[K_ind,ttm_ind],
                                                              impl_surf_2[K_ind,ttm_ind],
                                                              rho, T_v[ttm_ind], DF )

    return res_mat


cpdef spread_option_kirk_mat_simple (np.ndarray[np.float64_t, ndim=2] F_1_m, 
                                     np.ndarray[np.float64_t, ndim=2] F_2_m, 
                                     double K, 
                                     double T, 
                                     double sigma_1, 
                                     double sigma_2, 
                                     double rho,
                                     np.ndarray[np.float64_t, ndim=1] T_v, 
                                     np.ndarray[np.float64_t, ndim=1] DF_v):

    cdef int ttm_ind 
    cdef int K_ind 
    cdef double DF_u

    cdef np.ndarray[np.float64_t, ndim=2] res_mat = np.zeros ( (np.shape(F_1_m)[0], len(T_v)) )

    for ttm_ind in range(len(T_v)):
        print "ttm_idx ", ttm_ind
        for K_ind in range (np.shape(F_1_m)[0]):
            DF_u = DF_v[ttm_ind] 
            res_mat[K_ind,ttm_ind] = spread_option_kirk_fast (F_1_m[K_ind,ttm_ind], 
                                                              F_2_m[K_ind,ttm_ind], 
                                                              K, 
                                                              sigma_1,
                                                              sigma_2,
                                                              rho, T - T_v[ttm_ind], DF_u )

    return res_mat


cdef double d11_helper(double S_1, double S_2, double K, 
                       double sigma_1, double sigma_2, double rho, 
                       double T, 
                       double z):
  
    # accessory variables
    nu_1 = sigma_1 * sqrt(T)
    nu_2 = sigma_2 * sqrt(T)
    mu_1 = log(S_1) - 0.5 * sigma_1 **2 * T
    mu_2 = log(S_2) - 0.5 * sigma_2 **2 * T
    m_over = rho * z
    s_sqr = 1. - rho **2
    s_sqrt = sqrt(s_sqr)
    V = (log(exp(z * nu_2 + mu_2) + K) - mu_1)/nu_1

    cdf_int_1 = (m_over - V)/s_sqrt + s_sqrt*nu_1
    cdf_int_2 = cdf_int_1 - s_sqrt*nu_1
    H_1 = exp(mu_1 + m_over * nu_1 + 0.5 * s_sqr * nu_1* nu_1) * cdf(cdf_int_1)
    H_2 = (exp(z * nu_2 + mu_2) + K) * cdf(cdf_int_2)

    return H_1 - H_2


cpdef double spread_option_appx(double F_1, double F_2, double K, 
                                double sigma_1, double sigma_2, double rho, 
                                double T, double DF):

    cdef np.ndarray[np.float64_t, ndim=1] p_integ = \
        np.array([-3.66847085e+00,  -2.78329010e+00,   3.66847085e+00,
               -2.02594802e+00,  -1.32655708e+00,  -6.56809567e-01,
               -1.06611759e-16,   6.56809567e-01,   1.32655708e+00,
               2.78329010e+00,   2.02594802e+00])

    cdef np.ndarray[np.float64_t, ndim=1] w_integ = \
        np.array([1.43956039e-06,   3.46819466e-04,   1.43956039e-06,
               1.19113954e-02,   1.17227875e-01,   4.29359752e-01,
               6.54759287e-01,   4.29359752e-01,   1.17227875e-01,
               3.46819466e-04,   1.19113954e-02])

    cdef double s = 0.

    for idx in range(11):
        s += w_integ[idx] * d11_helper(F_1, F_2, K, sigma_1, sigma_2, rho, 
                                       T, p_integ[idx] * sqrt(2.))

    return DF * s / sqrt(np.pi)


def trivariate_spread_exact_integrat_fast2 (double X_3, double Y_2, np.ndarray[np.float64_t, ndim=1] F_v,
                                            double K, np.ndarray[np.float64_t, ndim=1] sigma_v,
                                            np.ndarray[np.float64_t, ndim=1] rho, double T, double DF ):
    cdef np.ndarray[np.float64_t, ndim=1] nu = sigma_v * sqrt (T)
    cdef np.ndarray[np.float64_t, ndim=1] mu = - 0.5 * nu ** 2 

    cdef double nu_1_d = nu[0] * sqrt (1 - rho[1]**2)
    cdef double nu_2_d = nu[1] * sqrt (1 - rho[2]**2)
    cdef double rho_Y1_Y2 = (rho[0] - rho[1]*rho[2])/sqrt (1-rho[1]**2) /sqrt (1 - rho[2]**2)
    cdef double sigma_Z = sqrt (1 - rho_Y1_Y2**2) 

    return black_greeks_fast ( F_v[0] * exp ( mu[0] + rho[1] * X_3 * nu[0] + rho_Y1_Y2 * Y_2 * nu_1_d + 0.5 * nu_1_d ** 2 * ( 1 - rho_Y1_Y2**2 ) ), \
                               K + F_v[2] * exp (X_3 * nu[2] + mu[2] ) + F_v[1] * exp (Y_2 * nu_2_d + mu[1] + rho[2] * X_3 * nu[1] ) , \
                               - log (DF) / T, nu_1_d * sqrt(1 - rho_Y1_Y2**2) / sqrt(T), T, 0)[0]  \
                               / ( 2 * np.pi ) * exp ( - ( X_3**2 + Y_2 **2) / 2.0 )



#
# computes the squared integral of samuelson behavior 
# \int _s ^t (e^{-B(T_i - u)} + sigma_L )^2 du 
#
cdef double sam_int_fast (double s, double t, double T_i, double beta, double sigma_L):
    """samuelson vol function"""
    cdef double t1 = exp (-2.0 * beta * (T_i-t) )/(2.0*beta) - exp(-2.0*beta*(T_i-s) ) / (2.0*beta)
    cdef double t2 = sigma_L**2 * (t-s)
    cdef double t3 = 2.0 * sigma_L /beta * ( exp (-beta*(T_i-t)) - exp (-beta *(T_i-s) ) )

    return sqrt((t1+t2+t3)/(t-s))

#
# helper function for apo_long function
#
cdef double A(double T_a, double T_b, double t, double t_i, double sigma_L, double beta):
    cdef double t1 = sigma_L **2 *(t_i-t)
    cdef double t2 = sigma_L * (exp(-beta * (T_b - t_i)) - exp(-beta * (T_b - t)) + \
                                exp(-beta * (T_a - t_i)) - exp(-beta * (T_a - t)))  / beta
    cdef double t3 = (exp(-2 * beta * ((T_a + T_b) / 2.0 - t_i)) - exp(-2 * beta * ((T_a + T_b) / 2.0 - t))) / (2*beta)
    return  (t1 + t2 + t3) / ((t_i - t ) * sam_int_fast(t, T_a, T_a, beta, sigma_L) * sam_int_fast(t, T_b, T_b, beta, sigma_L))


# compute skew strikes in the samuelson model 
# F_sim ... simulated forward prices
# ext_tenors ... extension tenors, which tenors from F_sim to take
# tau_i ... option expiry times
# T ... extension time
# K ... extension strike
# beta ... beta of the samuelson parameters
# sigma_L ... sigma of the samuelson parameters 
cpdef np.ndarray[np.float64_t, ndim=2] comp_skew_strikes(np.ndarray[np.float64_t, ndim=2] F_sim, 
                                                         np.ndarray[np.int_t, ndim=1] ext_tenors, 
                                                         np.ndarray[np.float64_t, ndim=1] tau_i, 
                                                         double T, double K, double beta, double sigma_L):
    cdef int i, sim 
    cdef np.ndarray[np.float64_t, ndim=2] K_skew = np.zeros ( (len(tau_i), np.shape(F_sim)[1] ), dtype=np.double )

    for i,tenor in zip (range(len(ext_tenors)), ext_tenors):
        for sim in range (np.shape (F_sim)[1]):
            K_skew[i,sim] = F_sim[tenor,sim] * exp( \
                sam_int_fast(0., tau_i[tenor], tau_i[tenor], beta, sigma_L) / \
                sam_int_fast(0., T, tau_i[tenor], beta, sigma_L) \
                * sqrt(tau_i[tenor] / T) \
                * log(np.average(F_sim[ext_tenors,sim]) / K))

    return K_skew

# compute skew vols
#cpdef np.ndarray[np.float64_t, ndim=2] comp_skew_vols (np.ndarray[np.float64_t, ndim=2] F_sim, 
#                                                       np.ndarray[np.float64_t, ndim=2] K_skew, 
#                                                       np.ndarray[np.float64_t, ndim=1] sigma_atm, 
#                                                       np.ndarray[np.float64_t, ndim=1] tau_i, 
#                                                       ):
    
#    for i in ext_tenors:
#        for sim in range (nb_sims): 
            # tau_i is WRONG WRONG WRONG WRONG 
            # sigma_skew = skew_fct ( tau_i[i], \ 
            #                        log (F_sim[i, sim] / K_skew[i,sim] ) / sigma_atm[i] / sqrt (tau_i[i] ) )
#            sigma_skew = 0
#    return sigma_skew 
    


# long APO calculation
# (same as APO long, just it is compiled)
# call_put_ind = 0 for CALL  -------------- NOT YET IMLEMENTED WRONG WRONG WRONG 
#                1 for PUT
# F_c ... futures curve
# sigma_c ... volatility curve
# rho_mat ... correlation between individual futures' contracts CHECK THIS CHECK THIS...
# Ti_c, ti_c ... future's maturity, option maturity for a futures' contract
# beta, sigma_L ... Samuelson params
# K ... apo strike price (CHECK THIS CHECK THIS)
# df ... discount factor until the APO maturity  CHECK CHECK CHECK
# T ... APO maturity CHECK CHECK CHECK
# cp_ind ... call/put indicator, 1.0 .... call, 0 ... put 
cpdef double apo_long_fast (np.ndarray[np.float64_t, ndim=1] F_c, double K, double df, double T,
                            np.ndarray[np.float64_t, ndim=1] sigma_c,
                            np.ndarray[np.float64_t, ndim=2] rho_mat,
                            np.ndarray[np.float64_t, ndim=1] Ti_c,
                            np.ndarray[np.float64_t, ndim=1] ti_c,
                            double t, double beta, double sigma_L, double cp_ind):

    cdef int N = len (F_c)
    cdef int i, j

    M_1 = np.average (F_c)
    
    M_2_term1 = sum (F_c**2 * np.array([exp (A(Ti_c[i], Ti_c[i], t, Ti_c[i], sigma_L, beta) \
                                          * sigma_c[i]**2 * (ti_c[i] - t) ) for i in range(N) ] ) )

    M_2_matrix_term = sum (np.array ([ 2 * F_c[i] * F_c[j] * \
                                    exp ( A( Ti_c[i], Ti_c[j], t, min (Ti_c[i], Ti_c[j]), sigma_L, beta ) * \
                                          rho_mat[i][j] * sigma_c[i] * sigma_c[j] * (ti_c[i] - t) ) \
                                    for i in range(N) for j in range(i+1, N ) ] ) \
                           )
    cdef double M_2 = (M_2_term1 + M_2_matrix_term ) / ( np.double (N**2) )

    return cp_ind * black_call_fast (M_1, K, -log (df)/T, sqrt ( log (M_2 / M_1 **2) ) / sqrt (T), T) + \
           (1. - cp_ind ) * black_put_fast (M_1, K, -log (df)/T, sqrt ( log (M_2 / M_1 **2) ) / sqrt (T), T)
