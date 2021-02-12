#
# tests for com_skew_tolling
import datetime

from unittest import TestCase

from mrds.tolling.com_skew_tolling import ComSkewTolling


class ComSkewTollingTest(TestCase):

    MKT_DATE = datetime.date(2015, 4, 1)

    def test_basic(self):
        """ Example of how to use ComSkewTolling class.
        """

        com_skew = ComSkewTolling.from_db(self.MKT_DATE
                                         , ['WTI', 'BRENT']
                                         , ['WTI', 'BRENT']
                                         , ['WTI', 'BRENT']
                                         , hours_partition = { 'WEEKDAY': [('WTI', 8) , ('BRENT', 16), ]
                                                             , 'WEEKEND': [('WTI', 16), ('BRENT', 8 ), ] })

        res1 = com_skew.simulate_spot_blocks( ('WTI', 'BRENT', )
                                            , 1000
                                            , datetime.date(2015, 5, 1)
                                            , datetime.date(2015, 6, 20))

        print(1+1)

        self.assertTrue(True)

