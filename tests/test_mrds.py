# test cases for the base mrds module.
# IMPORTANT: date has to be _before_ 2015-04-20, CHOOSE 2015-04-01 by default

import datetime
from unittest import TestCase

from mrds.mrds import ComSkew

MKT_DATE = datetime.date(2015, 4, 1)


class TestMrds(TestCase):

    MKT_DATE = MKT_DATE

    def test_from_db(self):
        """ Does ComSkew even work?
        """

        m1 = ComSkew.from_db(self.MKT_DATE, ['WTI'])

        self.assertTrue(True)

    def test_integr_analy(self):
        """ Tests the integrate analytircally method

        """

        m1 = ComSkew.from_db(self.MKT_DATE, ['WTI'])
        res = m1._ComSkew__integr_analy([1,2], -11, -22, 3, 4, 5, 6)

        self.assertTrue(True)

    def test_c_calibration(self):

        model = ComSkew.from_db(self.MKT_DATE, ['WTI'])
        res_1 = model._c_vec('WTI', datetime.date(2015, 8, 1))

        self.assertTrue(True)

    def test_simulate_curves_cpu(self):
        """ Tests whether the simulate curves actually runs on the cpu. This should always work.
        """

        m1         = ComSkew.from_db(self.MKT_DATE, ['WTI',])
        nb_sims    = 1000  # number of simulations
        sim_times  = [datetime.date(2015, 4, 20), datetime.date(2015, 5 , 1)]  # simulation times
        tenor_list = [datetime.date(2015, 8, 1) , datetime.date(2015, 12, 1)]  # tenors to simulate

        sim_curves = m1.simulate_curves( ['WTI'], nb_sims, sim_times, tenor_list = tenor_list )

        self.assertEqual(sim_curves['WTI'].shape, (len(sim_times), len(tenor_list), nb_sims) )

    def test_simulate_curves_cpu_more(self):
        """ Tests whether the simulate curves actually runs on the cpu. This should always work.
        """

        m1         = ComSkew.from_db(self.MKT_DATE, ['WTI',])
        nb_sims    = 1000  # number of simulations
        sim_times  = [datetime.date(2015, 8, 1), datetime.date(2015, 9 , 1), datetime.date(2015, 10, 1)]  # simulation times
        tenor_list = [datetime.date(2015, 11, 1) , datetime.date(2015, 12, 1), datetime.date(2016, 1, 1), datetime.date(2016, 2, 1), datetime.date(2016, 3, 1)]  # tenors to simulate

        sim_curves = m1.simulate_curves( ['WTI'], nb_sims, sim_times, tenor_list = tenor_list )

        self.assertEqual(sim_curves['WTI'].shape, (len(sim_times), len(tenor_list), nb_sims) )

    def test_simulate_curves_1nb(self):
        """ Tests whether the 1nb simulate curves.
        """

        m1 = ComSkew.from_db(self.MKT_DATE, ['WTI', ])
        nb_sims    = 1000  # number of simulations
        sim_times  = [datetime.date(2015, 4, 20), datetime.date(2015, 5 , 1)]  # simulation times

        res1 = m1.simulate_1nb( ['WTI', ], nb_sims, sim_times )

        self.assertIn('WTI', res1)


class TestComSkewMultiple(TestCase):
    """ Test for Com skew model for multiple assets.
    """

    MKT_DATE = MKT_DATE

    def test_calibrate_multiple(self):
        """ Tests whether the model even calibrates to multiple assets.
        """

        model = ComSkew.from_db(self.MKT_DATE, ('WTI', 'BRENT', ))
        res1 = model.simulate_curves( ('WTI', 'BRENT', )
                                    , 1000
                                    , [datetime.date(2015, 5, 1), datetime.date(2015, 6, 1), ]  # simulation times
                                    , [datetime.date(2015, 5, 20), datetime.date(2015, 7, 15), ]  # tenor idx to simulate
                                    )

        self.assertTrue(True)


TestMrds().test_c_calibration()
