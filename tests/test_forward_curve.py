
import datetime
from unittest import TestCase

from forward_curve import FwdCurve


class FwdCurveTest(TestCase):

    def test_fwd_curve_1(self):
        """ Elementary test if the forward curve even runs.

        """
        fwd_dates = [ datetime.date(2015, 5, 1)
                    , datetime.date(2015, 6, 1)
                    , datetime.date(2015, 7, 1)
                    , datetime.date(2015, 8, 1) ]
        fwd_values = [100., 110., 120., 130.]
        mkt_date   = datetime.date(2015, 4, 1)

        fc = FwdCurve(mkt_date, 'WTI2', fwd_dates, fwd_values)

        r1 = fc.fwd_value(datetime.date(2015, 5, 10))

        self.assertEqual(True, False)
