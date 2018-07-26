from mrds.mrds import mrds_calib, mrds_calib_multiple
import numpy as np
from unittest import TestCase, TestSuite

# IMPORTANT: date has to be _before_ 20150420


class TestMrds(TestCase):

    def test_simulate_curves_cuda(self):
        """
        Tests whether the simulate curves cuda actually runs.

        """

        m1 = mrds_calib('WTI', '20150401', 12)
        m1.update_sim_times([0.2, 0.4])
        m1.simulate_curves_cuda(10000)

    def test_spot_blocks_gen(self):
        m3 = mrds_calib_multiple( ['ATSI_2X16', 'ATSI-PEAK', 'ATSI_7X8', 'NG_MICHCON_GD-PEAK']
                                , '20150401'
                                , [9, 9, 9, 9]
                                , cuda_ind = True)
        m3.cash_vol_list = 5 * m3.atm_vol_list
        sp = m3.simulate_spot_blocks_all(50000
                                        , [[0, 1, 2, 3, 4], [5, 6]]
                                        , [[6, 18], [12, 12]]
                                        , cuda_ind=True)

        self.assertTrue(True)

    def test_simulate_curves_fom(self):
        """
        Tests the function simulate_curves_fom even runs.

        """

        m1 = mrds_calib('WTI', '20150401', 12)
        m1.update_sim_times(np.linspace(0.2, 1., 20))

        m1.simulate_curves_fom(0, 50000, cuda_ind=False)
        m1.simulate_curves_fom(0, 50000, cuda_ind=True)

        self.assertTrue(True)