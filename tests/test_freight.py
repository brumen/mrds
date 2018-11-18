# test for the freight model

import datetime, numpy as np, scipy.interpolate
from unittest import TestCase
from logging  import getLogger

import freight

logger = getLogger(__name__)

mktDate = datetime.date(2015, 4, 1)

# initial location of tankers.
N_init = { 'AMS': 3
         , 'NYC': 4
         , 'MIA': 1
         , 'LA' : 1
         , 'SHA': 8 }

fwd_curves = {'AMS': np.array([95., 96., 97., 98.]),
              'NYC': np.array([92., 93., 94., 95.]),
              'MIA': np.array([91., 92., 93., 94.]),
              'LA': np.array([90., 91., 95., 100.]),
              'SHA': np.array([85., 90., 95., 100.])}

fwd_dates = [ datetime.date(2015, 4, 1)
            , datetime.date(2015, 5, 1)
            , datetime.date(2015, 6, 1)
            , datetime.date(2015, 7, 1) ]
fwd_dates = {'AMS': fwd_dates,
             'NYC': fwd_dates,
             'MIA': fwd_dates,
             'LA' : fwd_dates,
             'SHA': fwd_dates }


def fwdFunction(mktDate: datetime.date, location : str, future_date : datetime.date, dcf = 365.25, fwdVol='fwd'):
    """
    Sample forward/vol function.

    :param:
    """

    diffs = [tenor - mktDate for tenor in (fwd_dates if fwdVol == 'fwd' else vol_dates)[location]]
    disc_tenors_numeric = np.array([float(elt.days) for elt in diffs])/dcf
    curve_numeric = scipy.interpolate.splrep(disc_tenors_numeric, (fwd_curves if fwdVol == 'fwd' else vol_curves)[location])

    return scipy.interpolate.splev((future_date - mktDate).days / dcf, curve_numeric)


vol_adder = 0.
vol_curves = {'AMS': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder,
              'NYC': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder,
              'MIA': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder,
              'LA' : np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder,
              'SHA': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder}
vol_dates = fwd_dates

# correlation matrix
corr_mtx = {('AMS', 'AMS'): 0.98,
            ('AMS', 'NYC'): 0.9,
            ('AMS', 'MIA'): 0.95,
            ('AMS', 'LA') : 0.99,
            ('AMS', 'SHA'): 0.8,
            ('NYC', 'NYC'): 0.98,
            ('NYC', 'MIA'): 0.97,
            ('NYC', 'LA') : 0.82,
            ('NYC', 'SHA'): 0.74,
            ('MIA', 'MIA'): 0.98,
            ('MIA', 'LA') : 0.89,
            ('MIA', 'SHA'): 0.91,
            ('LA', 'LA')  : 0.99,
            ('LA', 'SHA') : 0.90,
            ('SHA', 'SHA'): 0.98}

# amount of time to get from one location to the other
travel_mtx = {('AMS', 'NYC'): 1,
              ('AMS', 'MIA'): 1,
              ('AMS', 'LA') : 2,
              ('AMS', 'SHA'): 5,
              ('NYC', 'MIA'): 1,
              ('NYC', 'LA') : 3,
              ('NYC', 'SHA'): 6,
              ('MIA', 'LA') : 2,
              ('MIA', 'SHA'): 5,
              ('LA', 'SHA') : 3}

# how much it costs to transport between locations
cost_mtx = {('AMS', 'NYC'): 0.1,
            ('AMS', 'MIA'): 0.1,
            ('AMS', 'LA') : 0.2,
            ('AMS', 'SHA'): 0.5,
            ('NYC', 'MIA'): 0.1,
            ('NYC', 'LA') : 0.3,
            ('NYC', 'SHA'): 0.6,
            ('MIA', 'LA') : 0.2,
            ('MIA', 'SHA'): 0.5,
            ('LA', 'SHA') : 0.3 }


class FreightTest(TestCase):

    def test_xyz_locations(self):
        """
        Tests whether the X & Y generate the correct vector indices.
        """

        nbTimePeriods = 5
        freight1 = freight.Freight( mktDate
                                  , fwdFunction
                                  , lambda mktDate, location, futDate: fwdFunction(mktDate, location, futDate, fwdVol = 'vol')
                                  , corr_mtx
                                  , travel_mtx
                                  , cost_mtx
                                  , N_init
                                  , [mktDate + datetime.timedelta(days=30*idx) for idx in range(0,nbTimePeriods)])

        allIndices = []
        for i in range(freight1._nbLocations):
            for t in range(nbTimePeriods):
                allIndices.append(freight1._N(i,t))
                for j in range(freight1._nbLocations):
                    for u in range(t):
                        allIndices.append(freight1._X(i, j, u, t))
                        allIndices.append(freight1._Y(i, j, u, t))

        self.assertEqual(sorted(allIndices), list(range(5**2 * nbTimePeriods* (nbTimePeriods- 1) \
                                                        + nbTimePeriods * 5 )))  # 5 - number of locations

    def test_just_run(self):
        """
        Runs the test and reports results.

        """

        freight1 = freight.Freight( mktDate
                                  , fwdFunction
                                  , lambda mktDate, location, futDate: fwdFunction(mktDate, location, futDate, fwdVol = 'vol')
                                  , corr_mtx
                                  , travel_mtx
                                  , cost_mtx
                                  , N_init
                                  , [mktDate + datetime.timedelta(days=15*idx) for idx in range(0,10)])

        freight1.representHedge()  # this prints out the hedge

        self.assertTrue(True)
