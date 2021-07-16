# test for the freight model

import datetime
import numpy as np
import logging

from unittest import TestCase

from mrds.freight.freight import Freight
from mrds.forward_curve   import FwdCurve
from mrds.vols.vols       import ATMFVolatility

logger = logging.getLogger(__name__)


class FreightTest(TestCase):
    """ Class to test the Freight class.
    """

    mkt_date = datetime.date(2015, 4, 1)

    # initial location of tankers.
    N_init = { 'AMS': (mkt_date + datetime.timedelta(days=2), 3)
             , 'NYC': (mkt_date, 4)
             , 'MIA': (mkt_date, 1)
             , 'LA' : (mkt_date, 1)
             , 'SHA': (mkt_date + datetime.timedelta(days=5), 8) }

    locations = list(N_init.keys())  # possible locations

    fwd_curves = { 'AMS': [95., 96., 97., 98.]
                 , 'NYC': [92., 93., 94., 95.]
                 , 'MIA': [91., 92., 93., 94.]
                 , 'LA' : [90., 91., 95., 100.]
                 #  , 'SHA': np.array([105., 106., 107., 109.]) }
                 , 'SHA': [85., 90., 95., 100.] }

    fwd_curves_2 = { 'MIA': [97., 98., 99., 100.]
                   , 'NYC': [92., 93., 94., 95.]
                   , 'AMS': [91., 92., 93., 94.]
                   , 'LA' : [90., 91., 95., 100.]
                   , 'SHA': [85., 90., 95., 100.] }

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
    travel_times = {('AMS', 'NYC'): 9,
              ('AMS', 'MIA'): 14,
              ('AMS', 'LA') : 23,
              ('AMS', 'SHA'): 30,
              ('NYC', 'MIA'): 5,
              ('NYC', 'LA') : 14,
              ('NYC', 'SHA'): 25,
              ('MIA', 'LA') : 9,
              ('MIA', 'SHA'): 18,
              ('LA', 'SHA') : 9}

    # how much it costs to transport between locations
    cost_mtx = {('AMS', 'NYC'): 0.1,
            ('AMS', 'MIA'): 0.1,
            ('AMS', 'LA') : 0.2,
            ('AMS', 'SHA'): 0.5,
            ('NYC', 'MIA'): 0.1,
            ('NYC', 'LA') : 0.3,
            ('NYC', 'SHA'): 0.6,
            ('MIA', 'LA') : 0.2,
            ('MIA', 'SHA'): 0.6,
            ('LA', 'SHA') : 0.3 }

    def _simple_freight_object(self, nb_days = 30 ):
        """ Returns the simple freight object to be used later.
        """

        # construction of fwd_curves
        fwd_curves = { location: FwdCurve(self.mkt_date, location, self.fwd_dates[location], self.fwd_curves[location])
                       for location in self.locations }

        # construction of vol curves
        vol_curves = { location: ATMFVolatility( location
                                               , self.mkt_date
                                               , fwd_curves[location]
                                               , {vol_date: [vol_value]
                                                  for vol_date, vol_value in zip(self.vol_dates[location], self.vol_curves[location]) }
                                               , )
                       for location in self.locations }

        return Freight( self.mkt_date
                      , lambda location, fut_date: fwd_curves[location].fwd_value(fut_date)
                      , lambda location, fut_date: vol_curves[location].atm_vol(fut_date)
                      , self.corr_mtx
                      , self.travel_times
                      , self.cost_mtx
                      , self.N_init
                      , self.mkt_date + datetime.timedelta(days=nb_days) )

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

        nb_locations = len(list(self.N_init.keys()))

        self.assertEqual(sorted(allIndices), list(range(nb_locations**2 * nb_time_periods * (nb_time_periods - 1) \
                                                        + nb_time_periods * nb_locations )))

    def test_just_run_small(self):
        """ Runs the test and reports results.
        """

        freight_1 = self._simple_freight_object(nb_days = 30)

        # rh ... hedge representation.
        rh = freight_1.represent_hedge()
        print('LOCATIONS')
        Freight.pretty_dict(rh['locations'])
        print('MOVEMENTS COND')
        Freight.pretty_dict(rh['movements_cond'])
        print('MOVEMENTS UNCOND')
        Freight.pretty_dict(rh['movements_uncond'])
        Freight.pretty_dict(freight_1.show_dynamics())
        freight_1.show_dynamics_and_locations()

        self.assertTrue(True)

    def test_just_run_2(self):
        """ Runs the test and reports results.
        """

        freight_1 = self._simple_freight_object(nb_days=30)

        # rh ... hedge representation.
        rh = freight_1.represent_hedge()
        print('LOCATIONS')
        Freight.pretty_dict(rh['locations'])
        print('MOVEMENTS COND')
        Freight.pretty_dict(rh['movements_cond'])
        print('MOVEMENTS UNCOND')
        Freight.pretty_dict(rh['movements_uncond'])
        Freight.pretty_dict(freight_1.show_dynamics())
        freight_1.show_dynamics_and_locations()

        freight_1.fwd_curves = lambda mkt_date, location, fut_date: self.fwd_function( (self.fwd_curves_2[location], self.fwd_dates[location])
                                                                                     , mkt_date
                                                                                     , fut_date )

        rh = freight_1.represent_hedge()
        print('LOCATIONS')
        Freight.pretty_dict(rh['locations'])
        print('MOVEMENTS COND')
        Freight.pretty_dict(rh['movements_cond'])
        print('MOVEMENTS UNCOND')
        Freight.pretty_dict(rh['movements_uncond'])
        Freight.pretty_dict(freight_1.show_dynamics())
        freight_1.show_dynamics_and_locations()

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
        NYC is low freight
        """

        freight_1 = self._simple_freight_object(nb_days=10)

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


# FreightTest().test_just_run_small()
