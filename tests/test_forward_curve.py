import datetime

from unittest import TestCase

from mrds.forward_curve import FwdCurve


class FwdCurveTest(TestCase):

    MKT_DATE = datetime.date(2015, 4, 1)

    # TODO: FIX THESE TESTS.
    def test_fwd_curve_1(self):
        """ Elementary test if the forward curve even runs.

        """
        fwd_dates = [ datetime.date(2015, 5, 1)
                    , datetime.date(2015, 6, 1)
                    , datetime.date(2015, 7, 1)
                    , datetime.date(2015, 8, 1) ]
        fwd_values = [100., 110., 120., 130.]

        fc = FwdCurve(self.MKT_DATE, 'WTI2', fwd_dates, fwd_values)

        r1 = fc.fwd_value(datetime.date(2015, 5, 10))

        self.assertEqual(True, False)

    def test_fwd_curve_2(self):
        """ Elementary test if the forward curve even runs.
        """

        fc = FwdCurve.from_db(self.MKT_DATE, 'NG_MICHCON_GD-PEAK')

        r1 = fc.fwd_value(datetime.date(2015, 5, 10))

        self.assertEqual(True, False)
