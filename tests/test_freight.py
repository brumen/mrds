# test for the freight model

import datetime, numpy as np

import freight

from unittest import TestCase

mktDate = datetime.date(2015, 4, 1)  # '20150401'
T = 4  # nb. time periods

N_init = { 'AMS': 3
         , 'NYC': 4
         , 'MIA': 1
         , 'LA' : 1
         , 'SHA': 8 }  # initial distribution of tankers

fwd_curves = {'AMS': np.array([95., 96., 97., 98.]),
              'NYC': np.array([92., 93., 94., 95.]),
              'MIA': np.array([91., 92., 93., 94.]),
              'LA': np.array([90., 91., 95., 100.]),
              'SHA': np.array([85., 90., 95., 100.])}

fwdFunction = lambda mktDate, location, t: fwd_curves[location][0]  # some sample

vol_curves = {'AMS': np.array([0.3, 0.32, 0.35, 0.4]),
              'NYC': np.array([0.3, 0.32, 0.35, 0.4]),
              'MIA': np.array([0.3, 0.32, 0.35, 0.4]),
              'LA': np.array([0.3, 0.32, 0.35, 0.4]),
              'SHA': np.array([0.3, 0.32, 0.35, 0.4])}

volFunction = lambda mktDate, location, t: vol_curves[location][0]  # simple

# correlation matrix
corr_mtx = {('AMS', 'AMS'): 0.98,
            ('AMS', 'NYC'): 0.9,
            ('AMS', 'MIA'): 0.95,
            ('AMS', 'LA'): 0.99,
            ('AMS', 'SHA'): 0.8,
            ('NYC', 'NYC'): 0.98,
            ('NYC', 'MIA'): 0.97,
            ('NYC', 'LA'): 0.82,
            ('NYC', 'SHA'): 0.74,
            ('MIA', 'MIA'): 0.98,
            ('MIA', 'LA'): 0.89,
            ('MIA', 'SHA'): 0.91,
            ('LA', 'LA'): 0.99,
            ('LA', 'SHA'): 0.90,
            ('SHA', 'SHA'): 0.98}

# amount of time to get from one location to the other
travel_mtx = {('AMS', 'NYC'): 1,
              ('AMS', 'MIA'): 1,
              ('AMS', 'LA'): 2,
              ('AMS', 'SHA'): 5,
              ('NYC', 'MIA'): 1,
              ('NYC', 'LA'): 3,
              ('NYC', 'SHA'): 6,
              ('MIA', 'LA'): 2,
              ('MIA', 'SHA'): 5,
              ('LA', 'SHA'): 3}


class FreightTest(TestCase):

    def test_just_run(self):
        """
        Runs the test

        """

        freight1 = freight.Freight( mktDate
                                  , fwdFunction
                                  , volFunction
                                  , corr_mtx
                                  , travel_mtx
                                  , N_init
                                  , [mktDate + datetime.timedelta(days=15*idx) for idx in range(0,10)])

        # result = freight1.freightHedge()
        freight1.representHedge()  # this prints out the hedge

        self.assertTrue(True)
