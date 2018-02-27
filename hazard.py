# import openopt - optimization solver (from dev. version)
import openopt 
import numpy as np


def dp(T, Ti_v, h_v):
    """
    default probability function
    h_v longer than Ti_v by 1
    CHECK AGAIN CHECK AGAIN CHECK AGAIN
    """
    if len (Ti_v) == 0:
        return 1.0 - np.exp(- T * h_v[0])  # len (h_v) == 1
    else: 
        Ti_ind = Ti_v < T
        Ti_last = np.sum(Ti_ind)
        T_diff = np.diff(np.concatenate((np.array([0]), Ti_v)))
        return 1.0 - np.exp(-(np.sum(Ti_ind * T_diff * h_v[:-1]) + (Ti_last != 0) *
                              (T - Ti_v[Ti_last-1]) * h_v[Ti_last] +
                              (Ti_last == 0) * T * h_v[0]))


def cds_rate (R, T, Ti_v, h_v, DF):
    """
    cds rate for quarterly payments up till time T
    :param h_v: vector of hazard rates
    :param Ti_v: vector of times for those rates
    :param R: recovery rate
    :param T: 4*T has to be an integer
    """
    paydates = np.arange(1, round(T * 4.0) + 1)/4.0  # quarterly paydates
    paydates_diff = 0.25  # differences - quarters
    DF_paydates = np.array([DF(Ti) for Ti in paydates])
    DP_paydates = np.array([dp(Ti, Ti_v, h_v) for Ti in paydates])
    DP_paydates_diff = np.diff(np.array(np.concatenate((np.array([0]), DP_paydates))))
    return (1.0 - R) * np.sum(DF_paydates * (1. - np.array(np.concatenate((np.array([0]),
                                                                           DP_paydates[:-1])))) *
                              DP_paydates_diff)/\
           np.sum(DF_paydates * (1.0 - DP_paydates) * paydates_diff)
    

def solve_cds (cds_em_v, T_v, DF, params, h_init = 0.1, max_iter=50, iprint=-1, solver='scipy_cobyla'):
    """
    :param cds_em: empirical cds rates
    :param cds_an: analytical cds rates
    :param params: in a dictionary of the form ["R", "Ti_v"]
    :param R: recovery rate
    :param Ti_v: cutoff of recovery rates
    :param DF: discount function
    """
    R = params['R']
    Ti_v = params['Ti_v']
    h_v = np.zeros(len(Ti_v))

    for T_i, T_i_nb in zip (Ti_v, range(len(Ti_v))):
        optim_pr = openopt.NLP(lambda h: (cds_rate(R, T_i, Ti_v[:T_i_nb],
                                                   np.array(np.concatenate((h_v[:T_i_nb],
                                                                            np.array(h)))), DF) -
                                          cds_em_v[T_i_nb])**2 ,
                               h_init, lb=0., ub=np.inf, iprint=iprint) # , maxIter= max_iter, iprint=iprint )
        h_v[T_i_nb] = optim_pr.solve(solver).xf

    return h_v
