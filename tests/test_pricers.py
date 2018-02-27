import config
import numpy as np
import pycuda.gpuarray as gpa

import ds
import mrds
import pricers


def test_black_simple():
    print pricers.black_simple('20150401', 'WTI', '20151231', 50.)


def test_trivariate_spread(K=2.):
    """
    THIS TEST DOESNT WORK FOR ALL PARAMETERS
    """
    F_v = np.array([3., 2., 1.])
    sigma_v = np.array([0.2, 0.3, 0.4])
    rho = np.array([0.9, 0.95, 0.99])
    T = 1.
    df = 0.99

    print pricers.trivariate_spread_kirk(F_v, K, sigma_v, rho, T, df)
    print pricers.trivariate_spread_exact(F_v, K, sigma_v, rho, T, df)


def test_apo_vector():
    F_c_mat = 1


def test_trivariate_spread_exact():
    F_v = np.array([3., 2., 1.])
    sigma_v = np.array([0.2, 0.3, 0.4])
    rho = np.array([0.9, 0.95, 0.99])
    T = 1.
    df = 0.99

    #print "Q ", pricers.trivariate_spread_exact(F_v, 0.1, sigma_v, rho, T, df,
    #                                            quad_spgrid_ind='quad')

    print "SG", pricers.trivariate_spread_exact(F_v, 0.1, sigma_v, rho, T, df,
                                                quad_spgrid_ind='sg', sg_level=20)

