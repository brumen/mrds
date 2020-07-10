# Freight model implementation
#

import datetime
import numpy as np
import logging
from scipy.optimize import linprog
from functools      import lru_cache
from typing         import Dict, List, Callable, Tuple

from mrds.discount        import DiscountCurve
# TODO: FIX THIS BOTTOM LINE AND REMOVE THE FUNCTION FROM HERE.
# from mrds.pricers.pricers import spread_option_kirk


class FreightException(Exception):
    pass


logger = logging.getLogger(__name__)


def cdf_vec(x : np.array) -> np.array:
    """ Computes the cdf of the standard normal random variable of a vector x.
        Works for both vectors and matrices

    :param x: vector/matrix to compute the standard normal variable of.
    :returns: vector/matrix of results
    """

    l  = np.abs(x)
    k  = 1. / (1. + 0.2316419 * l)
    k2 = k**2
    k4 = k2**2

    # 0.39 = 1/sqrt(2*pi)
    w = 1. - 0.3989422804 * np.exp(-l*l / 2) * (0.31938153 * k -0.356563782 * k2 +
                                                1.781477937 * k2 * k + -1.821255978 * k4 + 1.330274429 * k4 * k)

    xPos = x >= 0.
    return w * xPos + (1. - w) * (~xPos)


def spread_option_kirk(F_1, F_2, K, sigma_1, sigma_2, rho, T, df):
    """ Kirk formula for bivariate spread option when strike K = 0
    TODO: IMPORT IT FROM mrds.pricers.pricers when this all works.

    """

    sigma_K = np.sqrt(sigma_1**2 - 2 * F_2 / (F_2 + K) * rho * sigma_1 * sigma_2 +
                      (F_2 / (F_2 + K)) ** 2 * sigma_2**2)
    d_1 = (np.log(F_1/(F_2+K)) + 0.5 * sigma_K**2 * T) / (sigma_K * np.sqrt(T))
    d_2 = d_1 - sigma_K * np.sqrt(T)

    return  df * (F_1 * cdf_vec(d_1) - (F_2+K) * cdf_vec(d_2))


class Freight:
    """ Freight class.
    """

    LARGE_NUMBER = 1000000.  # large number to prohibit travel between certain directions & times.

    def __init__(self
                 , mkt_date          : datetime.date
                 , fwd_curves        : Callable
                 , vol_curves        : Callable
                 , corr_matrix       : Dict[Tuple[str, str], float]
                 , travel_matrix     : Dict[Tuple[str, str], float]
                 , cost_matrix       : Dict[Tuple[str, str], float]
                 , initial_locations : Dict[str, int]
                 , time_grid         : List[datetime.date]
                 , dcf               : float = 365.25 ):
        """
        :param mkt_date: market date
        :param initial_locations: locations between which freight can be transported, list[str]
        :param fwd_curves: fucntion of (mkt_date, location, future date), returns forward curve for the future date.
        :param vol_curves: same as fwd_curves, except that it gives future volatility for that date
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

        self.mkt_date          = mkt_date
        self.initial_locations = initial_locations  # initial locations of the portfolio
        self._fwd_curves       = fwd_curves
        self._vol_curves       = vol_curves
        self._corr_matrix      = corr_matrix
        self._travel_matrix    = travel_matrix   # number of periods between different locations
        self._cost_matrix      = cost_matrix     # same as travel matrix, costs between locations
        self._time_grid        = time_grid       # grid used to compute the value of the freight portfolio.
        self._dcf              = dcf             # day count factor

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
        self.__discount_factor = None

    @property
    def mkt_date(self) -> datetime.date:
        return self._mkt_date

    @mkt_date.setter
    def mkt_date(self, new_mkt_date : datetime.date):
        self._mkt_date = new_mkt_date

    @property
    def fwd_curves(self):
        return self._fwd_curves

    @fwd_curves.setter
    def fwd_curves(self, new_fwd_curves):
        self._fwd_curves = new_fwd_curves

    @property
    def vol_curves(self):
        return self._vol_curves

    @vol_curves.setter
    def vol_curves(self, new_vol_curves):
        self._vol_curves = new_vol_curves

    def fwd_curves_curr(self, location: str, future_date: datetime.date) -> float:
        """ Gets the forward curves for market date for the times requested in timeList.

        :param location: location
        :param future_date: time for which the forward curve is requested
        :returns: forward rate for the future date.
        """

        return self.fwd_curves(self.mkt_date, location, future_date)

    def vol_curves_curr(self, location: str, future_date: datetime.date) -> float:
        """ Gets the volatility curves for market date for the times requested in timeList.

        :param location: location
        :param future_date: time for which the forward curve is requested
        :returns: forward volatility for the future date.
        """

        return self.vol_curves(self.mkt_date, location, future_date)

    @property
    def _discount_factor(self) -> Callable:
        """ Constructs the discount function for market date.

        @param mkt_date: market date of the discount function.
        """

        if self.__discount_factor:
            return self.__discount_factor

        self.__discount_factor = DiscountCurve.discount_function(self.mkt_date)  # this is a function
        return self.__discount_factor

    def DF(self, future_date : datetime.date) -> float:
        """ Computes the discount factor from self.mkt_date to future_date

        @param future_date: future date to which the discounting is computed
        @returns: discount factor from mkt date until future_date
        """

        return self._discount_factor(future_date)

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
                      , start_date : datetime.date
                      , end_date   : datetime.date ) -> float:
        """ Spread option value between city1, city2 and times start_date, end_date, start_date < end_date.

        :param city1: start location of the tanker
        :param city2: end location of the tanker
        :param start_date: start date for the travel between those locations
        :param end_date: end date for the travel between those locations.
        """

        return spread_option_kirk( self.fwd_curves_curr(city1, start_date)
                                 , self.fwd_curves_curr(city2, end_date)
                                 , self._cost_matrix[(city1, city2) if (city1, city2) in self._cost_matrix else (city2, city1)] if city1 != city2 else 0.
                                 , self.vol_curves_curr(city1, start_date)
                                 , self.vol_curves_curr(city2, end_date)
                                 , self._corr_matrix[(city1, city2) if (city1, city2) in self._corr_matrix else (city2, city1)]
                                 , (end_date - start_date).days / self._dcf
                                 , self.DF(end_date) )  # TODO: Check if this is correct here

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

    def _N(self, i : int, t : int) -> int:
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

    def __travel_allowed( self
                        , location_nb_1 : int
                        , location_nb_2 : int
                        , time_period_1 : int
                        , time_period_2 : int ) -> bool:
        """ Is travel between location_nb_1 and location_nb_2 allowed between times time_period_1 & time_period_2: time_period_2 > time_period_1

        :param location_nb_1: start location number of the tanker
        :param location_nb_2: destination location number of the tanker, e.g. 3 corresponds to self._nbs_to_locations[3]
        :param time_period_1: start time period of the tanker, e.g. 4
        :param time_period_2: end time period of the tanker, e.g. 7.
        """

        if location_nb_1 == location_nb_2:  # that is always allowed
            return True

        city_i, city_j = self._nbs_to_locations[location_nb_1], self._nbs_to_locations[location_nb_2]

        return time_period_2 - time_period_1 >= self._travel_matrix[(city_i, city_j) if (city_i, city_j) in self._travel_matrix else (city_j, city_i)]

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
        for location_nb in range(self._nb_locations):
            constraints_vec = np.zeros(self.__nb_lp_variables)
            constraints_vec[self._N(location_nb, 0)] = 1.
            equality_vector.append(self.initial_locations[self._nbs_to_locations[location_nb]])
            equality_matrix.append(constraints_vec)

        # sum_i n_i,t = K
        for time_period in range(1, self._nb_time_periods):  # t=0 already given above
            constraints_vec = np.zeros(self.__nb_lp_variables)
            for location_nb in range(self._nb_locations):
                constraints_vec[self._N(location_nb, time_period)] = 1.
            equality_vector.append(sum(self.initial_locations.values()))  # number of tankers, ships
            equality_matrix.append(constraints_vec)

        # constraint n_i,t = n_i,t-1 + sum_{j, u<t} (X(j, i, u, t) + Y(j,i,t,u)) - sum _{u>t-1, j} (X(i,j,t-1, u) + Y(i,j,t-1,u))
        for time_period_1 in range(1, self._nb_time_periods):
            for location_nb_1 in range(self._nb_locations):
                constraints_vec = np.zeros(self.__nb_lp_variables)
                constraints_vec[self._N(location_nb_1, time_period_1)]     =  1.
                constraints_vec[self._N(location_nb_1, time_period_1 - 1)] = -1.
                for location_nb_2 in range(self._nb_locations):
                    for time_period_2 in range(time_period_1):
                        constraints_vec[self._X(location_nb_2, location_nb_1, time_period_2, time_period_1)] = -1.
                        constraints_vec[self._Y(location_nb_2, location_nb_1, time_period_2, time_period_1)] = -1.
                    for time_period_2 in range(time_period_1, self._nb_time_periods):
                        constraints_vec[self._X(location_nb_1, location_nb_2, time_period_1 - 1, time_period_2)] = 1.
                        constraints_vec[self._Y(location_nb_1, location_nb_2, time_period_1 - 1, time_period_2)] = 1.
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

        for start_period in range(self._nb_time_periods):  # time period t = start_period
            for start_loc in range(self._nb_locations):  # cities  i = start_loc
                for end_loc in range(self._nb_locations):  # j = end_loc
                    for end_period in range(start_period+1, self._nb_time_periods):  # u = end_period
                        travel_allowed = self.__travel_allowed(start_loc, end_loc, start_period, end_period)
                        location_start = self._nbs_to_locations[start_loc]
                        location_end   = self._nbs_to_locations[end_loc]
                        start_time     = self._nbs_to_time_grid[start_period]
                        end_time       = self._nbs_to_time_grid[end_period]

                        self.__value_vec[self._X(start_loc, end_loc, start_period, end_period)] = - self._spread_option( location_start
                                                                                                                       , location_end
                                                                                                                       , start_time
                                                                                                                       , end_time ) if travel_allowed else self.LARGE_NUMBER
                        self.__value_vec[self._Y(start_loc, end_loc, start_period, end_period)] = -(  self.fwd_curves_curr(location_end  , end_time)
                                                                                                    - self.fwd_curves_curr(location_start, start_time)) \
                                                              if travel_allowed  else self.LARGE_NUMBER

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
                         , b_eq = ev )

        if result.success:
            self.__freight_hedge_result = result.x
            return self.__freight_hedge_result  # actual result

        raise FreightException(result.message)

    def freight_hedge_x(self, loc_1 : str, loc_2 : str, start_date: datetime.date, end_date: datetime.date):
        """ Displays the hedge for locations loc_1 and loc_2 between dates start_date and end_date.

        :param loc_1: start location, e.g. 'AMS'
        :param loc_2: end location, e.g. 'NYC'
        :param start_date: start of freight hedge between loc_1 and loc_2
        :param end_date: end of freight hedge between loc_1 and loc_2
        """

        return self.freight_hedge_x(self._X( self._locations.index(loc_1)
                                           , self._locations.index(loc_2)
                                           , self._time_grid.index(start_date)
                                           , self._time_grid.index(end_date) ) )

    def freight_hedge_y(self, loc_1 : str, loc_2 : str, start_date: datetime.date, end_date: datetime.date):

        return self.freight_hedge_y( self._Y( self._locations.index(loc_1)
                                            , self._locations.index(loc_2)
                                            , self._time_grid.index(start_date)
                                            , self._time_grid.index(end_date) ) )

    def _hedge_locations(self, ignore_small_nbs = .0001):
        """ Represents the hedge locations of the optimization problem.

        @param ignore_small_nbs: ignore hedges below this number
        """

        freight_hedge = self.freight_hedge

        locations = {}
        for period_nb in range(self._nb_time_periods):
            time_period = self._time_grid[period_nb]
            locations[time_period] = {}
            for location in self._locations:
                hedge = freight_hedge[self._N(self._locations.index(location), period_nb)]
                if hedge > ignore_small_nbs:
                    locations[time_period][location] = hedge
            if locations[time_period] == {}:
                locations.pop(time_period)

        return locations

    def _hedge_movements_cond_uncond(self, cond_uncond : str, ignore_small_nbs = 0.001):
        """ Conditional or unconditional representation of hedge movements.

        @param cond_uncond: 'cond' if conditional movments, else 'uncond'
        @param ignore_small_nbs: number below which the hedge positions are ignored.
        @returns: value of the conditional movements of freight. TODO: EXPLAIN THIS BETTER.
        """

        hedge           = self.freight_hedge
        cond_uncond_var = self._X if cond_uncond == 'cond' else self._Y

        hm_cond = {}
        for time_period in range(self._nb_time_periods):  # time_period was t
            grid_period = self._time_grid[time_period]
            hm_cond[grid_period] = {}
            for location in self._locations: # for i in range(self._nb_locations)}  # location was i = self._locations.index(location)
                i = self._locations.index(location)
                hm_cond[grid_period][location] = []
                for j in range(self._nb_locations):
                    for u in range(time_period + 1, self._nb_time_periods):
                        hedged_value = hedge[cond_uncond_var(i, j, time_period, u)]
                        if hedged_value > ignore_small_nbs:
                            hm_cond[grid_period][location].append((self._nbs_to_locations[j], self._time_grid[u], hedged_value))
                if hm_cond[grid_period][location] == []:
                    hm_cond[grid_period].pop(location)  # remove location
            if hm_cond[grid_period] == {}:
                hm_cond.pop(grid_period)

        return hm_cond

    @staticmethod
    def pretty_dict(d : Dict, indent : int = 0 ):
        """ Pretty print dictionary d.

        @param d: dictionary to be printed.
        @returns: prints the dict and returns None
        """

        for key, value in d.items():
            curr_key = '\t' * indent + str(key)
            if not isinstance(value, dict):
                print(curr_key + ': ' + str(value))
            else:
                print(curr_key)
                Freight.pretty_dict(value, indent+1)

    def represent_hedge(self, ignore_small_nbs = .0001 ):
        """ Represents the hedge obtained from optimization.

        @param ignore_small_nbs: ignore all hedges below a certain threshold, e.g. 1e-4
        """

        return { 'portfolioValue'  : - np.sum(self._value * np.array(self.freight_hedge))  # self._value is negative, cause linprog is minimized
               , 'locations'       : self._hedge_locations(ignore_small_nbs)
               , 'movements_cond'  : self._hedge_movements_cond_uncond('cond', ignore_small_nbs)
               , 'movements_uncond': self._hedge_movements_cond_uncond('uncond', ignore_small_nbs)
               , }
