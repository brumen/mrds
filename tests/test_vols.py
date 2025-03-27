#
# Volatility testing class

import datetime
import numpy as np

from unittest import TestCase

from mrds.forward_curve import FwdCurve
from mrds.vols.jwss7 import JWSS7Volatility, JWSS7VolatilityDisplay
from mrds.vols.quadratic import QuadraticVol
from mrds.data.quadratic_vols import wti2_vol
from mrds.vols.vols_basic import black_vol_inverse


class VolsBasicTest(TestCase):

    def test_black_vol_inverse(self):
        """ Tests the inverse black volatility.
        """

        dt = 1.
        DF = 0.99
        theta = 1
        tol = 1e-6

        for F in np.linspace(10., 1500., 200):
            for K in np.linspace(10., 1500., 200):
                for p in np.linspace(max(F-K, 0.) + 1.e-6, 5 * max(F-K, 0.) + 5.e-6, 100):
                    res = black_vol_inverse(F, K, p, dt, DF, theta, tol)

        # TODO: WHAT TO DO W/ THIS???

        self.assertTrue(True)


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


class C0C1C2Test(TestCase):

    def test_c0c1c2(self):

        mkt_date = datetime.date(2015, 4, 1)
        vol1 = QuadraticVol('WTI2'
                            , mkt_date
                            , FwdCurve.from_db( mkt_date, 'WTI')
                            , wti2_vol)

        self.assertTrue(False)

    def test_c0c1c2_fake(self):

        mkt_date = datetime.date(2015, 4, 1)
        vol1 = QuadraticVol.from_db('WTI'
                                    , mkt_date)

        self.assertTrue(False)
