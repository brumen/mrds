# Freight model implementation
#

import datetime
import numpy as np
import logging
from scipy.optimize import linprog

from typing import Dict, List, Callable

from mrds.discount import DF
from mrds.pricers.pricers import spread_option_kirk


class FreightException(Exception):
    pass


logger = logging.getLogger(__name__)


class Freight:
    """ Freight class.
    """

    LARGE_NUMBER = 1000000.  # large number to prohibit travel between certain directions & times.

    def __init__(self
                 , mkt_date          : datetime.date
                 , fwd_curve_fct     : Callable
                 , vol_curve_fct     : Callable
                 , corr_matrix       : Dict
                 , travel_matrix     : Dict
                 , cost_matrix       : Dict
                 , initial_locations : Dict
                 , time_grid         : List[datetime.date]
                 , dcf               : float = 365.25 ):
        """
        :param mkt_date: market date
        :param initial_locations: locations between which freight can be transported, list[str]
        :param fwd_curve_fct: fucntion of (location, mkt_date, future date), returns forward rate for that point.
        :param initial_locations: dictionary of how many ships are in a particular location.
                                 {location: nb_ships }
        :param corr_matrix: correlation between individual locations, dictionary where keys are
                           city pairs (city_1, city_2) and values are correlations between cities.
        :param travel_matrix: the amount of time it takes between different locations, a dictionary
                             where keys are location pairs (loc_1, loc_2) and values are time as fractions of
                             a year (i.e. 1. means 1 year).
        :param cost_matrix: same as travelMatrix, but refers to costs between cities.
        :param time_grid: time grid for the problem, i.e. time discretization on the movement of tankers.
        :param dcf: day count factor, used for discounting and option evaluation.
        """

        self.mkt_date       = mkt_date
        self.fwd_curve_fct  = fwd_curve_fct
        self.vol_curve_fct  = vol_curve_fct
        self._corr_matrix   = corr_matrix
        self._travel_matrix = travel_matrix   # number of periods between different locations
        self._cost_matrix   = cost_matrix     # same as travel matrix, costs between locations
        self._time_grid     = time_grid       # grid used to compute the value of the freight portfolio.
        self._dcf           = dcf            # day count factor
        self._initial_locations = initial_locations  # initial locations of the portfolio

        # simple derived variables
        self._locations         = list(initial_locations.keys())  # locations considered are given in initial_locations
        self._nb_locations      = len(self._locations)     # number of different locations
        self._nbs_to_locations  = {idx: loc for (idx, loc) in enumerate(self._locations)}
        self._nbs_to_time_grid = {idx: time_step for (idx, time_step) in enumerate(self._time_grid)}
        self._nb_time_periods   = len(self._time_grid)     # length of grid = number of time periods + 1

        # cached values, used as properties
        self.__value_vec   = None  # vector of all individual values
        self.__LM          = None
        self.__EM          = None
        self.__EV          = None
        self.__lower_bound = None
        self.__freight_hedge_result = None

    def fwd_vol_curves(self
                       , location : str
                       , future_date : datetime.date
                       , fwd_vol_ind ='fwd') -> float:
        """ Gets the forward curves for market date for the times requested in timeList.

        :param location: location, string
        :param future_date: time for which the forward curve is requested
        :param fwd_vol_ind: indicating whether 'fwd' or 'vol' is computed (string)
        :returns: array of forwards or vols for that timeList and location (returns vector)
        """

        return (self.fwd_curve_fct if fwd_vol_ind == 'fwd' else self.vol_curve_fct)(self.mkt_date, location, future_date)

    @property
    def lower_bound(self):
        """ Lower bound of the individual variables, which is 0, basically a vector of zeros.
        """

        if self.__lower_bound:
            return self.__lower_bound

        self.__lower_bound = np.zeros(self.__nb_lp_variables)

        return self.__lower_bound

    def _spread_option( self
                      , city1 : str
                      , city2 : str
                      , t1    : datetime.date
                      , t2    : datetime.date ) -> float:
        """ Spread option value between city1, city2 and times t1, t2, t1<t2.
        """

        return spread_option_kirk( self.fwd_vol_curves(city1, t1)
                                 , self.fwd_vol_curves(city2, t2)
                                 , self._cost_matrix[(city1, city2) if (city1, city2) in self._cost_matrix else (city2, city1)] if city1 != city2 else 0.
                                 , self.fwd_vol_curves(city1, t1, fwd_vol_ind='vol')
                                 , self.fwd_vol_curves(city2, t2, fwd_vol_ind='vol')
                                 , self._corr_matrix[(city1, city2) if (city1, city2) in self._corr_matrix else (city2, city1)]
                                 , (t2 - t1).days / self._dcf
                                 , DF(self.mkt_date, t2) )

    def _X(self, i : int, j : int, t : int, u :int) -> int:
        """ Conditional transport variable.
            Location of the variable x_(i,j,t,u) in the matrix, t<u
            Index corresponding to shipping from city i to city j between time t and u conditional.
        """

        return i + j * self._nb_locations + self._nb_locations ** 2 * (self._nb_time_periods - 1 - u + (self._nb_time_periods * t - t * (t + 1) // 2))

    def _Y(self, i : int, j: int, t : int, u: int) -> int:
        """ Unconditional transport variable.
        """

        # first line is the number of variables X
        return self._nb_locations ** 2 * self._nb_time_periods * (self._nb_time_periods - 1) // 2 \
               + self._X(i, j, t, u)

    def _N(self, i, t) -> int:
        """ Number of tankers in city i at time t. Location of the variable n_(i,t) in the vector of all variables.
        """

        return self._nb_locations ** 2 * self._nb_time_periods * (self._nb_time_periods - 1) \
               + i + t * self._nb_locations

    @property
    def __nb_lp_variables(self) -> int:
        """ Number of variables for the Linear problem. (X, Y, Z, N)
        """

        return self._nb_locations ** 2 * self._nb_time_periods * (self._nb_time_periods - 1) \
               + self._nb_time_periods * self._nb_locations

    def __travel_allowed(self, i : int, j : int, u : int, t : int) -> bool:
        """ Is travel between i and j allowed between times u & t: t> u
        """

        if i == j:  # that is always allowed
            return True

        city_i, city_j = self._nbs_to_locations[i], self._nbs_to_locations[j]

        return t - u >= self._travel_matrix[(city_i, city_j) if (city_i, city_j) in self._travel_matrix else (city_j, city_i)]

    @property
    def __lm_mat(self) -> np.array:
        """ Constructs the inequality matrix self.__LM, i.e. for conditions:
               self.__LM * x <= 0. (0. is a vector)
        """

        if self.__LM is not None:
            return self.__LM

        constraints_mat = []  # constraint matrix

        # constraint n_{i,t} >= sum_{j,u} X(i,j,t,u) + Y(i,j,t,u)
        for t in range(self._nb_time_periods):
            for i in range(self._nb_locations):
                constraints_vec = np.zeros(self.__nb_lp_variables)
                constraints_vec[self._N(i, t)] = -1.
                for j in range(self._nb_locations):
                    for u in range(t+1, self._nb_time_periods):
                        constraints_vec[self._X(i, j, t, u)] = 1.
                        constraints_vec[self._Y(i, j, t, u)] = 1.
                constraints_mat.append(constraints_vec)

        self.__LM = np.array(constraints_mat)

        return self.__LM

    @property
    def __EMV_mat(self) -> (np.array, np.array):
        """ Sets the equality matrix (EM) and equality vector (EV), for constraints:
              EM * x = EV, EM is a matrix, EV is a vector.
        """

        if self.__EM is not None and self.__EV is not None:
            return self.__EM, self.__EV

        equality_matrix = []
        equality_vector = []

        # initial setting of N: N(i,0) = initialLocation(i)
        for i in range(self._nb_locations):
            constraints_vec = np.zeros(self.__nb_lp_variables)
            constraints_vec[self._N(i, 0)] = 1.
            equality_vector.append(self._initial_locations[self._nbs_to_locations[i]])
            equality_matrix.append(constraints_vec)

        # sum_i n_i,t = K 
        for t in range(1, self._nb_time_periods):  # t=0 already given above
            constraints_vec = np.zeros(self.__nb_lp_variables)
            for i in range(self._nb_locations):
                constraints_vec[self._N(i, t)] = 1.
            equality_vector.append(sum(self._initial_locations.values()))  # number of tankers, ships
            equality_matrix.append(constraints_vec)

        # constraint n_i,t = n_i,t-1 + sum_{j, u<t} (X(j, i, u, t) + Y(j,i,t,u)) - sum _{u>t-1, j} (X(i,j,t-1, u) + Y(i,j,t-1,u))
        for t in range(1, self._nb_time_periods):
            for i in range(self._nb_locations):
                constraints_vec = np.zeros(self.__nb_lp_variables)  # constraints_vec is constraints vector
                constraints_vec[self._N(i, t)]     =  1.
                constraints_vec[self._N(i, t - 1)] = -1.
                for j in range(self._nb_locations):
                    for u in range(t):
                        constraints_vec[self._X(j, i, u, t)] = -1.
                        constraints_vec[self._Y(j, i, u, t)] = -1.
                    for u in range(t, self._nb_time_periods):
                        constraints_vec[self._X(i, j, t - 1, u)] = 1.
                        constraints_vec[self._Y(i, j, t - 1, u)] = 1.
                equality_matrix.append(constraints_vec)
                equality_vector.append(0.)

        self.__EM = np.array(equality_matrix)
        self.__EV = np.array(equality_vector)

        return self.__EM, self.__EV

    @property
    def _value(self):
        """ Setting the valuation vector - this is set for _MINIMIZING (COST) FUNCTION, therefore the values are negative.
        """

        if self.__value_vec is not None:
            return self.__value_vec

        self.__value_vec = np.zeros(self.__nb_lp_variables)

        for t in range(self._nb_time_periods):  # time period
            for i in range(self._nb_locations):  # cities
                for j in range(self._nb_locations):
                    for u in range(t+1, self._nb_time_periods):
                        self.__value_vec[self._X(i, j, t, u)] = - self._spread_option(self._nbs_to_locations[i]
                                                                                      , self._nbs_to_locations[j]
                                                                                      , self._nbs_to_time_grid[t]
                                                                                      , self._nbs_to_time_grid[u]) if self.__travel_allowed(i, j, t, u) else self.LARGE_NUMBER
                        self.__value_vec[self._Y(i, j, t, u)] = -(self.fwd_vol_curves(self._nbs_to_locations[j], self._nbs_to_time_grid[u])
                                                                  - self.fwd_vol_curves(self._nbs_to_locations[i], self._nbs_to_time_grid[t])) \
                                                              if self.__travel_allowed(i, j, t, u) else self.LARGE_NUMBER

        return self.__value_vec

    @property
    def freight_hedge(self):
        """ Find optimum freight hedge, solve the linear program.
        """

        if self.__freight_hedge_result is not None:
            return self.__freight_hedge_result

        em, ev = self.__EMV_mat  # equality matrix condition em * x = ev

        result = linprog(self._value
                         , A_ub  = self.__lm_mat  # inequality condition A_ub * x <= b_ub
                         , b_ub = np.zeros(self.__lm_mat.shape[0])  # zeros the shape of LMMat
                         , A_eq = em
                         , b_eq = ev)

        if result.success:
            self.__freight_hedge_result = result.x
            return self.__freight_hedge_result  # actual result

        raise FreightException(result.message)

    def __freight_hedge_x(self, i : int, j : int, t : int, u : int):
        return self.freight_hedge[self._X(i, j, t, u)]

    def freight_hedge_x(self, loc_1 : str, loc_2 : str, start_date: datetime.date, end_date: datetime.date):
        """ Displays the hedge for locations loc_1 and loc_2 between dates start_date and end_date.
        """

        return self.__freight_hedge_x( self._locations.index(loc_1)
                                     , self._locations.index(loc_2)
                                     , self._time_grid.index(start_date)
                                     , self._time_grid.index(end_date) )

    def __freight_hedge_y(self, i, j, t, u):
        return self.freight_hedge[self._Y(i, j, t, u)]

    def freight_hedge_y(self, loc_1 : str, loc_2 : str, start_date: datetime.date, end_date: datetime.date):

        return self.__freight_hedge_y( self._locations.index(loc_1)
                                     , self._locations.index(loc_2)
                                     , self._time_grid.index(start_date)
                                     , self._time_grid.index(end_date) )

    def represent_hedge(self):
        """ Represents the hedge obtained from optimization.
        """

        hedge = self.freight_hedge

        locations = {self._time_grid[t]: {self._nbs_to_locations[i]: hedge[self._N(i, t)] for i in range(self._nb_locations)}
                     for t in range(self._nb_time_periods)}

        movements_cond = {self._time_grid[t]: {self._nbs_to_locations[i]: [(self._nbs_to_locations[j], self._time_grid[u], hedge[self._X(i, j, t, u)])
                                                                           for j in range(self._nb_locations) for u in range(t + 1, self._nb_time_periods)]
                                               for i in range(self._nb_locations)}
                          for t in range(self._nb_time_periods)}

        movements_uncond = {self._time_grid[t]: {self._nbs_to_locations[i]: [(self._nbs_to_locations[j], self._time_grid[u], hedge[self._Y(i, j, t, u)])
                                                                             for j in range(self._nb_locations) for u in range(t + 1, self._nb_time_periods)]
                                                 for i in range(self._nb_locations)}
                            for t in range(self._nb_time_periods)}

        return { 'portfolioValue': - np.sum(np.array(self._value) * np.array(hedge))  # self.valueVec is negative, cuase linprog is minimized
               , 'locations'     : locations
               , 'movements'     : (movements_cond, movements_uncond)}
