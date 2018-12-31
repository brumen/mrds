#
# Volatility testing class

import datetime

from unittest import TestCase, TestLoader, TextTestRunner

from vols.vols import JW7Volatility


class JWVolTest(TestCase):

    def test_just_run(self):
        """
        Tries to run the JW7Volatility class.

        """

        vol1 = JW7Volatility.fromDb('WTI', datetime.date(2015, 4, 1))

        self.assertTrue(True)


def main():
    """
    Run the tests.

    """

    TextTestRunner(verbosity=2).run(TestLoader().loadTestsFromTestCase(JWVolTest))
