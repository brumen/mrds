# test cases for the base mrds module.

import datetime
import numpy as np

from mrds import ComSkew
from mrds_utils import mrds_calib, mrds_calib_multiple
from unittest   import TestCase

# IMPORTANT: date has to be _before_ 20150420


class TestMrds(TestCase):

    def test_from_db(self):
        m1 = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI'])

        self.assertTrue(True)

    def test_simulate_curves_cpu(self):
        """
        Tests whether the simulate curves actually runs on the cpu. This should always work.

        """

        m1 = mrds_calib('WTI', '20150401', 12)
        m1.simulation_times = [0.2, 0.4]
        m1.simulate_curves(10000, cuda_ind=False)

        self.assertTrue(True)

    def test_simulate_curves_cuda(self):
        """
        Tests whether the simulate curves cuda actually runs.

        """

        m1 = mrds_calib('WTI', '20150401', 12)
        m1.simulation_times = [0.2, 0.4]
        m1.simulate_curves(10000, cuda_ind=True)

        self.assertTrue(True)

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
        m1.simulation_times = np.linspace(0.2, 1., 20)

        m1.simulate_curves_fom(0, 50000, cuda_ind = False)
        m1.simulate_curves_fom(0, 50000, cuda_ind = True )

        self.assertTrue(True)

    def test_gen_spot_rn_cpu(self):
        """
        Tests whether the simulate curves actually runs on the cpu. This should always work.

        """

        m1 = mrds_calib('WTI', '20150401', 12)
        m1.simulation_times = [0.2, 0.4]
        print(m1.simulate_spot(0, 100))

        self.assertTrue(True)
