#
# Volatility testing class

import datetime

from unittest import TestCase, TestLoader, TextTestRunner

from vols.vols import JWSS7Volatility


class JWVolTest(TestCase):

    def test_just_run(self):
        """
        Tries to run the JW7Volatility class.

        """

        vol1 = JWSS7Volatility.fromDb('WTI', datetime.date(2015, 4, 1))
        vol1._transform_from_jwss7(datetime.date(2015, 1, 10))
        vol1.implied_vol(datetime.date(2015, 1, 10), 100., 1.)

        self.assertTrue(True)


def main():
    """
    Run the tests.

    """

    TextTestRunner(verbosity=2).run(TestLoader().loadTestsFromTestCase(JWVolTest))
