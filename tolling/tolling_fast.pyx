# Fast version of the profit function 
import config
from numpy import *
cimport numpy as np


# running profit from power plant 
cpdef double running_profit_fast(double F_1, double F_2):
    return F_1 - 0.2 * F_2 - 0.2 # 0.2 IS SOME ADDITIONAL COSTS 

cpdef tensor_fast(np.ndarray[np.float64_t, ndim=4] P_m,
                  np.ndarray[np.float64_t, ndim=2] H_m,
                  np.ndarray[np.float64_t, ndim=2] G_m,
                  np.ndarray[np.float64_t, ndim=2] res_m):

    cdef int F_1_ind
    cdef int F_2_ind
    
    for F_1_ind in range(P_m.shape[0]):
        for F_2_ind in range(P_m.shape[1]):
            res_m[F_1_ind,F_2_ind] = sum (P_m[F_1_ind,F_2_ind,:,:] * H_m) + G_m[F_1_ind,F_2_ind]


cpdef tensor_fast_mat(np.ndarray[np.float64_t, ndim=4] P_m,
                      np.ndarray[np.float64_t, ndim=2] H_m,
                      np.ndarray[np.float64_t, ndim=2] G_m,
                      np.ndarray[np.float64_t, ndim=2] res_m):

    cdef int F_1_ind, F_2_ind, F_1_out, F_2_out

    for F_1_ind in range(P_m.shape[0]):
        for F_2_ind in range(P_m.shape[1]):
            res_m[F_1_ind,F_2_ind] = 0.0
            for F_1_out in range (P_m.shape[0]):
                for F_2_out in range (P_m.shape[1]):
                    res_m[F_1_ind,F_2_ind] += P_m[F_1_ind,F_2_ind,F_1_out,F_2_out] * H_m[F_1_out,F_2_out]
            res_m[F_1_ind,F_2_ind] = res_m[F_1_ind,F_2_ind] + G_m[F_1_ind,F_2_ind]
