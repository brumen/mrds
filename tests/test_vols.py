#
# Volatility testing class

import datetime

from unittest import TestCase, TestLoader, TextTestRunner

from forward_curve   import FwdCurve
from vols.vols       import JWSS7Volatility, C0C1C2Volatility
from data.c0c1c2vols import wti2_vol


class JWVolTest(TestCase):

    def test_just_run(self):
        """ Tries to run the JW7Volatility class.
        """

        vol1 = JWSS7Volatility.from_db('WTI', datetime.date(2015, 4, 1))
        vol1._transform_from_jwss7(datetime.date(2015, 1, 10))
        vol1.implied_vol(datetime.date(2015, 1, 10), 100., 1.)

        self.assertTrue(True)


class C0C1C2Tet(TestCase):

    def test_c0c1c2(self):

        mkt_date = datetime.date(2015, 4, 1)
        vol1 = C0C1C2Volatility( 'WTI2'
                               , mkt_date
                               , FwdCurve.from_db( mkt_date, 'WTI')
                               , wti2_vol )

        self.assertTrue(False)
