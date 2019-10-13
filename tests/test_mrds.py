# test cases for the base mrds module.

import datetime
from unittest import TestCase

from mrds import ComSkew

# IMPORTANT: date has to be _before_ 20150420


class TestMrds(TestCase):

    def test_from_db(self):
        """ Does ComSkew even work?
        """

        m1 = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI'])

        self.assertTrue(True)

    def test_simulate_curves_cpu(self):
        """ Tests whether the simulate curves actually runs on the cpu. This should always work.
        """

        m1 = ComSkew.from_db(datetime.date(2015, 4, 1), ['WTI',])
        m1.simulation_times = [0.2, 0.4]
        m1.simulate_curves(10000, cuda_ind=False)

        self.assertTrue(True)
