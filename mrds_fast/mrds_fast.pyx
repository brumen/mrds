# import config 
import numpy as np
cimport numpy as np

# declarations of external functions 
cdef extern from "math.h":
    double exp (double)


# returns the 4-dim tensor (i, j, n, m)
# for k-l correlation 
def corr_fast(np.ndarray[np.float64_t, ndim=2] rho, 
              np.ndarray[np.float64_t, ndim=1] beta_k,
              np.ndarray[np.float64_t, ndim=1] beta_l, 
              np.ndarray[np.float64_t, ndim=1] sigma_k, 
              np.ndarray[np.float64_t, ndim=1] sigma_l,
              np.ndarray[np.float64_t, ndim=1] kappa_k, 
              np.ndarray[np.float64_t, ndim=1] kappa_l,
              np.ndarray[np.float64_t, ndim=1] T_v_i, 
              np.ndarray[np.float64_t, ndim=1] T_v_j,
              double t):

    cdef np.ndarray[np.float64_t, ndim=4] G = np.np.zeros((len(beta_k), len(beta_l),
                                                        len (kappa_k), len (kappa_l)),
                                                       dtype=np.double)
    cdef int i, j, n, m
    cdef np.ndarray[np.int32_t, ndim=1] b_k_ran = np.arange(len(beta_k), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] b_l_ran = np.arange(len(beta_l), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] k_k_ran = np.arange(len(kappa_k), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] k_l_ran = np.arange(len(kappa_l), dtype=np.int32)
    
    for i in b_k_ran:
        for j in b_l_ran:
            for n in k_k_ran:
                for m in k_l_ran:
                    G[i, j, n, m] = rho[n, m] * beta_k[i] * beta_l[j] * sigma_k[n] * sigma_l[m] * \
                                    exp(- kappa_k[n] * T_v_i[i] - kappa_l[m] * T_v_j[j] ) * (exp((kappa_k[n] + kappa_l[m]) *t) - 1.) / \
                                    (kappa_k[n] + kappa_l[m])

    return G


# returns the 4-dim tensor (i, j, n, m)
# for k-l correlation 
def corr_fast_sum(np.ndarray[np.float64_t, ndim=2] rho, 
                  np.ndarray[np.float64_t, ndim=1] beta_k,
                  np.ndarray[np.float64_t, ndim=1] beta_l, 
                  np.ndarray[np.float64_t, ndim=1] sigma_k, 
                  np.ndarray[np.float64_t, ndim=1] sigma_l,
                  np.ndarray[np.float64_t, ndim=1] kappa_k, 
                  np.ndarray[np.float64_t, ndim=1] kappa_l,
                  np.ndarray[np.float64_t, ndim=1] T_v_i, 
                  np.ndarray[np.float64_t, ndim=1] T_v_j,
                  double t):

    cdef int i, j, n, m
    cdef np.ndarray[np.float64_t, ndim=2] Gsum = np.zeros((len(beta_k), len(beta_l)), dtype=np.double)
    cdef np.ndarray[np.int32_t, ndim=1] b_k_ran = np.arange(len(beta_k), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] b_l_ran = np.arange(len(beta_l), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] k_k_ran = np.arange(len(kappa_k), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] k_l_ran = np.arange(len(kappa_l), dtype=np.int32)

    for i in b_k_ran:
        for j in b_l_ran:
            Gsum[i,j] = 0.
            for n in k_k_ran:
                for m in k_l_ran:
                    Gsum[i,j] += rho[n, m] * beta_k[i] * beta_l[j] * sigma_k[n] * sigma_l[m] * \
                                 exp(- kappa_k[n] * T_v_i[i] - kappa_l[m] * T_v_j[j] ) * (exp((kappa_k[n] + kappa_l[m]) *t) - 1.) / \
                                 (kappa_k[n] + kappa_l[m])

    return Gsum



# Gkl has to be given, too slow to generate it 
# G = corr_fast_sum (rho, beta_k, beta_l, sigma_k, sigma_l, kappa_k, kappa_l, T_v_i, T_v_j, t)
# Gkl = G[k,l]
# a0_k, a1_k, a2_k, a3_k, a4_k, V_k = a_k
# a0_l, a1_l, a2_l, a3_l, a4_l, V_l = a_l 
def curve_corr_skew(int k, l,
                    double Gkl, 
                    np.ndarray[np.float64_t, ndim=1] a_k,
                    np.ndarray[np.float64_t, ndim=1] a_l,
                    np.ndarray[np.float64_t, ndim=2] rho, 
                    np.ndarray[np.float64_t, ndim=1] beta_k,
                    np.ndarray[np.float64_t, ndim=1] beta_l, 
                    np.ndarray[np.float64_t, ndim=1] sigma_k, 
                    np.ndarray[np.float64_t, ndim=1] sigma_l,
                    np.ndarray[np.float64_t, ndim=1] kappa_k, 
                    np.ndarray[np.float64_t, ndim=1] kappa_l,
                    np.ndarray[np.float64_t, ndim=1] T_v_i, 
                    np.ndarray[np.float64_t, ndim=1] T_v_j,
                    double t):
    
    # compute correlation
    cdef double sum1 = a_k[0] * a_l[0] + a_k[0] * a_l[2] * a_l[5] + a_k[2] * a_l[0] * a_k[5]
    cdef double sum2 = a_k[0] * a_l[4] * 3. * a_l[5] **2 + a_k[4] * a_l[0] * 3. * a_k[5] **2
    cdef double sum3 = a_k[1] * a_l[1] * Gkl
    cdef double sum4 = a_k[1] * a_l[3] * Gkl * a_l[5] + a_k[3] * a_l[1] * Gkl * a_k[5]
    cdef double sum5 = a_k[2] * a_l[2] * Gkl**2 * 3. * a_k[5]**2
    cdef double sum6 = a_k[2] * a_l[4] * Gkl**2 * 15. * a_k[5]**3 + a_k[4] * a_l[2] * Gkl**2 * 15. * a_l[5]**3
    cdef double sum7 = a_k[4] * a_l[4] * Gkl**4 * 105. * a_k[5]**4

    return sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 
