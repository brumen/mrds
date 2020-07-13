# test for the freight model

import datetime
import numpy as np
import scipy.interpolate
import logging

from unittest import TestCase

from mrds.freight.freight         import Freight
from mrds.freight.freight_display import FreightDisplay

logger = logging.getLogger(__name__)



class FreightTest(TestCase):
    """ Class to test the Freight class.
    """

    mkt_date = datetime.date(2015, 4, 1)

    # initial location of tankers.
    N_init = { 'AMS': 3
             , 'NYC': 4
             , 'MIA': 1
             , 'LA' : 1
             , 'SHA': 8 }

    fwd_curves = { 'AMS': np.array([95., 96., 97., 98.])
                 , 'NYC': np.array([92., 93., 94., 95.])
                 , 'MIA': np.array([91., 92., 93., 94.])
                 , 'LA' : np.array([90., 91., 95., 100.])
                 , 'SHA': np.array([85., 90., 95., 100.]) }

    fwd_dates_d = [ datetime.date(2015, 4, 1)
                  , datetime.date(2015, 5, 1)
                  , datetime.date(2015, 6, 1)
                  , datetime.date(2015, 7, 1) ]

    # future dates when the forward prices are given.
    fwd_dates = { 'AMS': fwd_dates_d
                , 'NYC': fwd_dates_d
                , 'MIA': fwd_dates_d
                , 'LA' : fwd_dates_d
                , 'SHA': fwd_dates_d }

    def fwd_function( self
                    , mkt_date      : datetime.date
                    , location      : str
                    , future_date   : datetime.date
                    , dcf           : float = 365.25
                    , fwd_vol       : str   = 'fwd' ):
        """ Sample forward/vol function.

        :param mkt_date: market date for which forwards/vols are given
        :param location: location for the forward curve
        :param future_date: future date for which forward is desired
        :param dcf: day-count factor
        :param fwd_vol: whether forward or vol curve is obtained.
        """

        diffs = [tenor - mkt_date for tenor in (self.fwd_dates if fwd_vol == 'fwd' else self.vol_dates)[location]]
        disc_tenors_numeric = np.array([float(elt.days) for elt in diffs])/dcf
        curve_numeric = scipy.interpolate.splrep(disc_tenors_numeric, (self.fwd_curves if fwd_vol == 'fwd' else self.vol_curves)[location])

        return scipy.interpolate.splev((future_date - mkt_date).days / dcf, curve_numeric)

    # volatility part of the model.
    vol_adder = 0.
    vol_curves = {  'AMS': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder
                  , 'NYC': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder
                  , 'MIA': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder
                  , 'LA' : np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder
                  , 'SHA': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder }
    vol_dates = fwd_dates

    # correlation matrix
    corr_mtx = { ('AMS', 'AMS'): 0.98
                 ,  ('AMS', 'NYC'): 0.9
                 , ('AMS', 'MIA'): 0.95
                 , ('AMS', 'LA') : 0.99
                 , ('AMS', 'SHA'): 0.8
                 , ('NYC', 'NYC'): 0.98
                 , ('NYC', 'MIA'): 0.97
                 , ('NYC', 'LA') : 0.82
                 , ('NYC', 'SHA'): 0.74
                 , ('MIA', 'MIA'): 0.98
                 , ('MIA', 'LA') : 0.89
                 , ('MIA', 'SHA'): 0.91
                 , ('LA', 'LA')  : 0.99
                 , ('LA', 'SHA') : 0.90
                 , ('SHA', 'SHA'): 0.98}

    # amount of time to get from one location to the other
    travel_times = {('AMS', 'NYC'): 1,
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

    def _simple_freight_object(self, nb_time_periods = 5):
        """ Returns the simple freight object to be used later.
        """

        return Freight( self.mkt_date
                      , self.fwd_function
                      , lambda mkt_date, location, futDate: self.fwd_function(mkt_date, location, futDate, fwd_vol ='vol')
                      , self.corr_mtx
                      , self.travel_times
                      , self.cost_mtx
                      , self.N_init
                      , [self.mkt_date + datetime.timedelta(days=30*idx) for idx in range(0, nb_time_periods)])

    def test_xyz_locations(self):
        """ Tests whether the X & Y generate the correct vector indices.
        """

        freight_1 = self._simple_freight_object()

        allIndices = []
        nb_locations = len(freight_1.initial_locations)
        nb_time_periods = freight_1._nb_time_periods
        for i in range(nb_locations):
            for t in range(nb_time_periods):
                allIndices.append(freight_1._N(i, t))
                for j in range(nb_locations):
                    for u in range(t):
                        allIndices.append(freight_1._X(i, j, u, t))
                        allIndices.append(freight_1._Y(i, j, u, t))

        self.assertEqual(sorted(allIndices), list(range(5**2 * nb_time_periods * (nb_time_periods - 1) \
                                                        + nb_time_periods * 5 )))  # 5 - number of locations

    def test_just_run(self):
        """ Runs the test and reports results.
        """

        freight_1 = Freight( self.mkt_date
                           , self.fwd_function
                           , lambda mkt_date, location, fut_date: self.fwd_function(mkt_date, location, fut_date, fwd_vol ='vol')
                           , self.corr_mtx
                           , self.travel_times
                           , self.cost_mtx
                           , self.N_init
                           , [self.mkt_date + datetime.timedelta(days=15*idx) for idx in range(0,10)])

        # rh ... hedge representation.
        rh = freight_1.represent_hedge()
        print('LOCATIONS')
        Freight.pretty_dict(rh['locations'])
        print('MOVEMENTS COND')
        Freight.pretty_dict(rh['movements_cond'])
        print('MOVEMENTS UNCOND')
        Freight.pretty_dict(rh['movements_uncond'])
        Freight.pretty_dict(freight_1.show_dynamics())

        self.assertTrue(True)

    def test_freight_display(self):
        """ Demonstrates the usage of the display class.
        """

        self._simple_freight_object().display_movement()

        self.assertTrue(True)


class SmallFreightTest(FreightTest):
    """ Class to test the Freight dispatch in a small setting.
    """

    mkt_date = datetime.date(2015, 4, 1)

    # initial location of tankers.
    N_init = { 'AMS': 3
             , 'NYC': 0 }

    fwd_curves = { 'AMS': np.array([95., 96., 97., 98.])
                 , 'NYC': np.array([92., 93., 94., 95.]) }

    fwd_dates_d = [ datetime.date(2015, 4, 1)
                  , datetime.date(2015, 5, 1)
                  , datetime.date(2015, 6, 1)
                  , datetime.date(2015, 7, 1) ]

    # future dates when the forward prices are given.
    fwd_dates = { 'AMS': fwd_dates_d
                , 'NYC': fwd_dates_d }

    # volatility part of the model.
    vol_adder = 0.
    vol_curves = {  'AMS': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder
                  , 'NYC': np.array([0.3, 0.32, 0.35, 0.4]) + vol_adder }
    vol_dates = fwd_dates

    # correlation matrix
    corr_mtx = { ('AMS', 'AMS'): 0.98
               , ('AMS', 'NYC'): 0.9
               , ('NYC', 'NYC'): 0.95
               , }

    # amount of time to get from one location to the other
    travel_times = { ('AMS', 'NYC'): 1
                   , }

    # how much it costs to transport between locations
    cost_mtx = {('AMS', 'NYC'): 0.1, }

    def test_just_run(self):
        """ Runs the test and reports results.
        NYC is low freigth
        """

        freight_1 = Freight( self.mkt_date
                           , self.fwd_function
                           , lambda mkt_date, location, fut_date: self.fwd_function(mkt_date, location, fut_date, fwd_vol ='vol')
                           , self.corr_mtx
                           , self.travel_times
                           , self.cost_mtx
                           , self.N_init
                           , [self.mkt_date + datetime.timedelta(days=15*idx) for idx in range(0,10)])

        # rh ... hedge representation.
        rh = freight_1.represent_hedge()
        print('VALUE')
        print(rh['portfolio_value'])
        print('LOCATIONS')
        Freight.pretty_dict(rh['locations'])
        print('MOVEMENTS COND')
        Freight.pretty_dict(rh['movements_cond'])
        print('MOVEMENTS UNCOND')
        Freight.pretty_dict(rh['movements_uncond'])
        Freight.pretty_dict(freight_1.show_dynamics())

        self.assertTrue(True)
