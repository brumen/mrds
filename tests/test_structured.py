import config
import numpy as np
import pycuda.gpuarray as gpa

import mrds
import structured


def test_sb():
    nb_sim = 300000
    mm = mrds.mrds_calib('WTI', '20150401', 12)
    mm.update_sim_times(np.linspace(0.01, 0.9, 100))
    mm.simulate_curves(nb_sim, tenor_list=[[11]])
    print [structured.sb(mm, [K, 70.]) for K in np.linspace(50., 70., 21)]


def test_sb_cuda():
    nb_sim = 5000000
    mm = mrds.mrds_calib('WTI', '20150401', 12)
    mm.update_sim_times(np.linspace(0.01, 0.9, 50))
    mm.simulate_curves(nb_sim, tenor_list=[[11]])
    sc = mm.simulated_curves[0]
    sc_d = gpa.to_gpu(sc).astype(np.float32)
    print [structured.sb_cuda(mm, sc_d, [K, 70.]) for K in np.linspace(50., 70., 21)]


def test_sb_cuda_2():
    nb_sim = 300000
    mm = mrds.mrds_calib('WTI', '20150401', 12)
    mm.update_sim_times(np.linspace(0.01, 0.9, 100))
    mm.simulate_curves_cuda(nb_sim, tenor_list=[[11]])
    sc = mm.simulated_curves[0]
    # paths_d - cols are sim times, rows are repetitions
    paths_d = sc[:, 0, :]
    payoff_d = sc[-1, 0, :]
    disc = mm.DF(mm.simulation_times[-1])

    print [structured.sb_cuda_fast(payoff_d, paths_d, [K, 70.], disc)
           for K in np.linspace(50., 70., 21)]


def test_db():
    nb_sim = 500000
    mm = mrds.mrds_calib('WTI', '20150401', 12)
    mm.update_sim_times(np.linspace(0.01, 0.9, 20))
    mm.simulate_curves(nb_sim, tenor_list=[[11]])
    # parameters
    params = [[40., 80.], 50.]

    for K in np.linspace(50., 70., 21):
        params[-1] = K
        print structured.db(mm, params)


def test_db_cuda():
    nb_sim = 1000000
    mm = mrds.mrds_calib('WTI', '20150401', 12)
    mm.update_sim_times(np.linspace(0.01, 0.9, 20))
    mm.simulate_curves_cuda(nb_sim, tenor_list=[[11]])
    # parameters
    params = [[40., 80.], 50.]

    for K in np.linspace(50., 70., 21):
        params[-1] = K
        print structured.db_cuda(mm, params)

