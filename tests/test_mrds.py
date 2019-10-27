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

        m1 = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI',])
        m1.simulate_curves( ['WTI']
                          , 1000
                          , [0.2, 0.4]
                          , tenor_list = [datetime.date(2015, 8, 1), datetime.date(2015, 12, 1)] )

        self.assertTrue(True)
