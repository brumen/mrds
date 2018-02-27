import config
import mrds
import numpy as np
import time

# date has to be _before_ 20150420
# m1 = mrds.mrds_calib('WTI', '20150401', 12)
# m2 = mrds.mrds_calib_multiple(['ATSI-PEAK', 'ATSI-2X16'], '20150401', [12, 12])


def test_simulate_curves_cuda():
    """
    tests whether the simulate curves cuda works correctly
    """
    m1 = mrds.mrds_calib('WTI', '20150401', 12)
    m1.update_sim_times([0.2, 0.4])
    m1.simulate_curves_cuda(10000)
    # print "SIM", m1.simulated_curves


def test_spot_blocks_gen(ci=True):
    m3 = mrds.mrds_calib_multiple(['ATSI_2X16', 'ATSI-PEAK', 'ATSI_7X8', 'NG_MICHCON_GD-PEAK'],
                                   '20150401', [9, 9, 9, 9],
                                   cuda_ind=ci)
    m3.cash_vol_list = 5 * m3.atm_vol_list
    print m3.simulate_spot_blocks_all(50000,
                                      [[0, 1, 2, 3, 4], [5, 6]],
                                      [[6, 18], [12, 12]], cuda_ind=ci)


def test_simulate_curves_fom(ci=False):
    """
    tests the function simulate_curves_fom
    """
    m1 = mrds.mrds_calib('WTI', '20150401', 12)
    m1.update_sim_times(np.linspace(0.2, 1., 20))
    t1 = time.time()
    for i in range(100):
        m1.simulate_curves_fom(0, 50000, cuda_ind=False)
    print "Time cpu:", time.time() - t1

    t1 = time.time()
    for i in range(100):
        m1.simulate_curves_fom(0, 50000, cuda_ind=True)
    print "Time cuda:", time.time() - t1
