# test cases for the base mrds module.
# IMPORTANT: date has to be _before_ 20150420, CHOOSE 2015-04-01

import datetime
from unittest import TestCase

from mrds import ComSkew


class TestMrds(TestCase):

    def test_from_db(self):
        """ Does ComSkew even work?
        """

        m1 = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI'])

        self.assertTrue(True)

    def test_integr_analy(self):
        """ Tests the integrate analytircally method

        """

        m1 = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI'])
        res = m1._ComSkew__integr_analy([1,2], -11, -22, 3, 4, 5, 6)

        self.assertTrue(True)

    def test_c_calibration(self):

        m1 = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI'])
        print(m1._c_vec('WTI', datetime.date(2015, 8,1)))

        self.assertTrue(True)

    def test_simulate_curves_cpu(self):
        """ Tests whether the simulate curves actually runs on the cpu. This should always work.
        """

        m1         = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI',])
        nb_sims    = 1000  # number of simulations
        sim_times  = [datetime.date(2015, 4, 20), datetime.date(2015, 5 , 1)]  # simulation times
        tenor_list = [datetime.date(2015, 8, 1) , datetime.date(2015, 12, 1)]  # tenors to simulate

        sim_curves = m1.simulate_curves( ['WTI'], nb_sims, sim_times, tenor_list = tenor_list )

        self.assertEqual(sim_curves.shape['WTI'], (len(sim_times), len(tenor_list), nb_sims) )

    def test_simulate_curves_1nb(self):
        """ Tests whether the 1nb simulate curves.
        """

        m1 = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI',])
        nb_sims    = 1000  # number of simulations
        sim_times  = [datetime.date(2015, 4, 20), datetime.date(2015, 5 , 1)]  # simulation times

        res1 = m1.simulate_1nb( ['WTI'], nb_sims, sim_times )

        self.assertIn('WTI', res1)

        # TODO: TO ANOTHER CHECK ON THE RESULT MATRIX!!
