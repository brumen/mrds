#
# Volatility testing class

import datetime

from unittest import TestCase

from forward_curve   import FwdCurve
from vols.jwss7      import JWSS7Volatility, JWSS7VolatilityDisplay
from vols.c0c1c2     import C0C1C2Volatility
from data.c0c1c2vols import wti2_vol


class JWVolTest(TestCase):

    def test_just_run(self):
        """ Tries to run the JW7Volatility class.
        """

        vol1 = JWSS7Volatility.from_db('WTI', datetime.date(2015, 4, 1))
        vol1._transform_from_jwss7(datetime.date(2015, 1, 10))
        vol1.implied_vol(datetime.date(2015, 1, 10), 100., 1.)

        self.assertTrue(True)

    def test_jwss7_display(self):
        """ Tests whether the volatility runs at all.
        """

        vol_1 = JWSS7VolatilityDisplay.from_db('WTI', datetime.date(2015, 4, 1))
        vol_1.draw_surface(datetime.date(2015, 6, 1), (70., 120., 10.), (0.5, 1.5, 0.1))

        self.assertTrue(True)


class C0C1C2Tet(TestCase):

    def test_c0c1c2(self):

        mkt_date = datetime.date(2015, 4, 1)
        vol1 = C0C1C2Volatility( 'WTI2'
                               , mkt_date
                               , FwdCurve.from_db( mkt_date, 'WTI')
                               , wti2_vol )

        self.assertTrue(False)

    def test_c0c1c2_fake(self):

        mkt_date = datetime.date(2015, 4, 1)
        vol1 = C0C1C2Volatility.from_db( 'WTI'
                                       , mkt_date )

        self.assertTrue(False)

JWVolTest().test_jwss7_display()
