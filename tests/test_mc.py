import config
import numpy as np
# import pycuda.gpuarray as gpa

import mc
import vols
import air_option as ao


def test_0(nb_sim=100):
    F_d = np.zeros(5)+ 100.
    s_d = np.zeros(5) + 0.2
    rho_m = vols.corr_hyp_sec_mat(0.99, range(5)).astype(np.float32)
    v1 = mc.mc_mult_steps(F_d, s_d, [0.2, 0.3], rho_m, nb_sim=nb_sim)
    return v1


def test_1(nb_sim=100):
    F_d = gpa.zeros(5, dtype=np.float32) + 100.
    s_d = gpa.zeros(5, dtype=np.float32) + 0.2
    rho_m = vols.corr_hyp_sec_mat(0.99, range(5)).astype(np.float32)
    v1 = mc.mc_mult_steps_cuda(F_d, s_d, [0.2, 0.3], rho_m, nb_sim=nb_sim)
    return v1


def test_2():
    F = np.array([100.])
    K = 100.
    T_end = 2.
    nb_sim = 1048575 # 2**20 - 1
    T = np.array ([T_end - 9./12., T_end - 6./12., T_end - 5./12.])
    s = np.array([0.25])
    rho_m = np.array([[1.]])
    df = 0.99
    T_l = np.linspace (0.1,1., 10)
    F_sim_l = mc.mc_mult_steps(F, s, T_l, rho_m, nb_sim)
    print np.average (F_sim_l[-1,:,0])


def test_3():
    F_v = (np.array([100., 105., 106.]), np.array([200., 205.]))
    s_v = (np.array([0.2, 0.2, 0.2]), np.array([0.2, 0.3]))
    T_l = ([0.1, 0.2, 0.3, 0.4, 0.5], [0.15, 0.25, 0.35, 0.45, 0.55])
    rho_m = (vols.corr_hyp_sec_mat(0.95, range(3)), vols.corr_hyp_sec_mat(0.95, range(2)))
    nb_sim = 1000
    ao_p = {'model': 'max',
            'F_max_prev': np.zeros((2, nb_sim)),
            'K': 200.,
            'penalty': 100., 
            'P_arg_max': 0.}

    res = mc.mc_mult_steps_cpu_ret(F_v, s_v, T_l, rho_m, nb_sim,
                                   ao_f=ao.ao_f_arb,
                                   ao_p=ao_p,
                                   d_v=None, cva_vals=None, model='ln')
    print "r", res['F_max_prev']


def test_4():
    """
    tests for the improvement of gpu version 
    """
    F_v = (np.array([100., 105., 106.]),
           np.array([200., 205.]))
    s_v = (np.array([lambda x: 0.2 for i in range(3)]),
           np.array([lambda x: 0.2 for i in range(2)]))
    d_v = (np.array([lambda x: 0.4 for i in range(3)]),
           np.array([lambda x: 0.2 for i in range(2)]))
    T_l_num = (np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
               np.array([0.15, 0.25, 0.35, 0.45, 0.55]))
    T_mat_num = (np.array([0.61, 0.62, 0.63]),
                 np.array([0.815, 0.825]))
    # rho_m = (vols.corr_hyp_sec_mat(0.95, range(3)), vols.corr_hyp_sec_mat(0.95, range(2)))
    rho_m = 0.95  # vols.corr_hyp_sec_mat(0.95, range(3)), vols.corr_hyp_sec_mat(0.95, range(2)))
    nb_sim = 10000
    ao_p = {'model': 'max',
            'F_max_prev': np.zeros((2, nb_sim)),
            'K': 200.,
            'penalty': 100., 
            'P_arg_max': 0.}

    return ao.compute_option_raw(F_v, s_v, T_l_num, T_mat_num, 200., 0.2, 0.2,
                                 rho_m, nb_sim=10000,
                                 d_v=d_v,
                                 model='max', cuda_ind=True,
                                 underlyer='n',
                                 gen_first=True)
