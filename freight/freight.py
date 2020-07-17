# Freight model implementation
#

import copy
import datetime
import numpy as np
import logging

from scipy.optimize import linprog
from functools      import lru_cache
from typing         import Dict, List, Callable, Tuple

from mrds.discount  import DiscountCurve
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
                 , travel_times      : Dict[Tuple[str, str], float]
                 , cost_matrix       : Dict[Tuple[str, str], float]
                 , initial_locations : Dict[str, Tuple[datetime.date, int]]
                 , time_grid         : List[datetime.date]
                 , dcf               : float = 365.25 ):
        """
        :param mkt_date: market date
        :param fwd_curves: fucntion of (mkt_date, location, future date), returns forward curve for the future date.
        :param vol_curves: same as fwd_curves, except that it gives future volatility for that date
        :param initial_locations: dictionary of how many ships are in a particular location at which date, can be in the future
                                 {location: (future_date, nb_ships) }
        :param corr_matrix: correlation between individual locations, dictionary where keys are
                           city pairs (city_1, city_2) and values are correlations between cities.
        :param travel_times: the amount of time it takes between different locations, a dictionary
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
        self._travel_times     = travel_times   # number of periods between different locations
        self._cost_matrix      = cost_matrix    # same as travel matrix, costs between locations
        self._time_grid        = time_grid      # grid used to compute the value of the freight portfolio.
        self._dcf              = dcf            # day count factor

        # cached values, used as properties
        self.__discount_factor = None  # discount factor function
        self.__freight_hedge   = None  # transport schedule
        self.__transport_value = None  # value of transport option
        self.__recompute_hedge = True  # whether to recompute the hedge, in response to the setting of the new curves

    @property
    def mkt_date(self) -> datetime.date:
        return self._mkt_date

    @mkt_date.setter
    def mkt_date(self, new_mkt_date : datetime.date):
        self.__recompute_hedge = True
        self._mkt_date = new_mkt_date

    @property
    def fwd_curves(self):
        return self._fwd_curves

    @fwd_curves.setter
    def fwd_curves(self, new_fwd_curves):
        self.__recompute_hedge = True
        self._fwd_curves = new_fwd_curves

    @property
    def vol_curves(self):
        return self._vol_curves

    @vol_curves.setter
    def vol_curves(self, new_vol_curves):
        self.__recompute_hedge = True
        self._vol_curves = new_vol_curves

    @property
    def _locations(self) -> List[str]:
        """ Locations used in the optimization problem.
        """

        return list(self.initial_locations.keys())

    @property
    def _nb_locations(self) -> int:
        """ Number of different locations.
        """

        return len(self._locations)

    @property
    def _nb_time_periods(self) -> int:
        """ Number of time periods used in the optimization problem.

        length of grid = number of time periods + 1
        """

        return len(self._time_grid)

    @property
    def _nbs_to_locations(self) -> Dict[int, str]:
        """ Each location is assigned a respective number. This is a mapping between the two.
        """

        return {idx: loc for (idx, loc) in enumerate(self._locations)}

    @property
    def _nbs_to_time_grid(self) -> Dict[int, int]:
        return {idx: time_step for (idx, time_step) in enumerate(self._time_grid)}


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

    def _cost_btw_locs(self, start_loc : str, end_loc : str ) -> float:
        """ Cost function between locations

        :param start_loc: start location, e.g. 'MIA'
        :param end_loc: end location, e.g. 'AMS'
        :returns: cost between these two locations
        """

        if start_loc == end_loc:
            return 0.

        # cost has to be in the cost matrix.
        assert (start_loc, end_loc) in self._cost_matrix or (end_loc, start_loc) in self._cost_matrix, \
            '({0}, {1}) not in cost matrix: {2}'.format(start_loc, end_loc, list(self._cost_matrix.keys()))

        # actual cost between cities
        return self._cost_matrix[(start_loc, end_loc) if (start_loc, end_loc) in self._cost_matrix else (end_loc, start_loc)]

    @lru_cache(maxsize=1000)
    def _spread_option( self
                      , market_date : datetime.date
                      , start_loc   : str
                      , end_loc     : str
                      , start_date  : datetime.date
                      , end_date    : datetime.date ) -> float:
        """ Spread option value between start_loc, end_loc and times start_date, end_date, start_date < end_date.

        :param start_loc: start location of the tanker
        :param end_loc: end location of the tanker
        :param start_date: start date for the travel between those locations
        :param end_date: end date for the travel between those locations.
        """

        return spread_option_kirk( self.fwd_curves_curr(start_loc, start_date)
                                 , self.fwd_curves_curr(end_loc  , end_date  )
                                 , self._cost_btw_locs(start_loc, end_loc)
                                 , self.vol_curves_curr(start_loc, start_date)
                                 , self.vol_curves_curr(end_loc, end_date)
                                 , self._corr_matrix[(start_loc, end_loc) if (start_loc, end_loc) in self._corr_matrix else (end_loc, start_loc)]
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

        return time_period_2 - time_period_1 >= self._travel_times[(city_i, city_j) if (city_i, city_j) in self._travel_times else (city_j, city_i)]

    def __lm_mat(self) -> np.array:
        """ Constructs the inequality matrix self.__LM, i.e. for conditions:
               self.__LM * x <= 0. (0. is a vector)
        """

        constraints_mat = []  # constraint matrix

        # constraint n_{i,t} >= sum_{j,u} X(i,j,t,u) + Y(i,j,t,u)
        for time_step_start in range(self._nb_time_periods):
            for location_start  in range(self._nb_locations):
                constraints_vec = np.zeros(self.__nb_lp_variables())
                constraints_vec[self._N(location_start, time_step_start)] = -1.
                for location_end in range(self._nb_locations):
                    for time_step_end in range(time_step_start + 1, self._nb_time_periods):
                        constraints_vec[self._X(location_start, location_end, time_step_start, time_step_end)] = 1.
                        constraints_vec[self._Y(location_start, location_end, time_step_start, time_step_end)] = 1.
                constraints_mat.append(constraints_vec)

        return np.array(constraints_mat)

    def __find_datetime_in_time_grid(future_date : datetime.date) -> int:
        return self._time_grid.index(future_date)

    def __EMV_mat(self) -> Tuple[np.array, np.array]:
        """ Sets the equality matrix (EM) and equality vector (EV), for constraints:
              EM * x = EV, EM is a matrix, EV is a vector.

        @returns: tuple of equality matrix and equality vector.
        """

        equality_matrix = []
        equality_vector = []

        # initial setting of N: N(i,0) = initialLocation(i)
        # N(i, initial_days) = initial_locations[i]
        for location, (initial_date, nb_tankers) in self.initial_locations.items():
            constraints_vec = np.zeros(self.__nb_lp_variables())
            location_nb = self._locations.index(location)
            initial_date_index_in_time_grid = self._time_grid.index(initial_date)
            constraints_vec[self._N(location_nb, initial_date_index_in_time_grid)] = 1.
            # TODO: THIS LINE BELOW IS WRONG
            # equality_vector.append(self.initial_locations[self._nbs_to_locations[location_nb]])
            equality_vector.append(nb_tankers)
            equality_matrix.append(constraints_vec)
        # for location_nb in range(self._nb_locations):
        #     constraints_vec = np.zeros(self.__nb_lp_variables())
        #     constraints_vec[self._N(location_nb, 0)] = 1.
        #     equality_vector.append(self.initial_locations[self._nbs_to_locations[location_nb]])
        #     equality_matrix.append(constraints_vec)

        # TODO: CHECK IF THIS IS OK HERE!!! MAYBE YOU CAN OMIT IT.
        # sum_i n_i,t = K
        # for time_period in range(1, self._nb_time_periods):  # t=0 already given above
        #     constraints_vec = np.zeros(self.__nb_lp_variables())
        #     for location_nb in range(self._nb_locations):
        #         constraints_vec[self._N(location_nb, time_period)] = 1.
        #     equality_vector.append(sum(self.initial_locations.values()))  # number of tankers, ships
        #     equality_matrix.append(constraints_vec)

        # constraint n_i,t = n_i,t-1 + sum_{j, u<t} (X(j, i, u, t) + Y(j,i,t,u)) - sum _{u>t-1, j} (X(i,j,t-1, u) + Y(i,j,t-1,u))
        for time_period_1 in range(1, self._nb_time_periods):
            for location_nb_1 in range(self._nb_locations):
                constraints_vec = np.zeros(self.__nb_lp_variables())
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

        return np.array(equality_matrix), np.array(equality_vector)

    def _value(self):
        """ Setting the valuation vector - this is set for _MINIMIZING (COST) FUNCTION, therefore the values are negative.
        """

        value_vec = np.zeros(self.__nb_lp_variables())
        mkt_date  = self.mkt_date

        for start_period in range(self._nb_time_periods):  # time period t = start_period
            for start_loc in range(self._nb_locations):  # cities  i = start_loc
                for end_loc in range(self._nb_locations):  # j = end_loc
                    for end_period in range(start_period+1, self._nb_time_periods):  # u = end_period
                        travel_allowed = self.__travel_allowed(start_loc, end_loc, start_period, end_period)
                        location_start = self._nbs_to_locations[start_loc]
                        location_end   = self._nbs_to_locations[end_loc]
                        start_time     = self._nbs_to_time_grid[start_period]
                        end_time       = self._nbs_to_time_grid[end_period]
                        costs_btw      = self._cost_btw_locs(location_start, location_end)

                        value_vec[self._X(start_loc, end_loc, start_period, end_period)] = - self._spread_option( mkt_date
                                                                                                                , location_start
                                                                                                                , location_end
                                                                                                                , start_time
                                                                                                                , end_time ) if travel_allowed else self.LARGE_NUMBER
                        # either a full tanker, or just the cost of transporting it.
                        value_vec[self._Y(start_loc, end_loc, start_period, end_period)] = - max(  self.fwd_curves_curr(location_end  , end_time)
                                                                                                  - self.fwd_curves_curr(location_start, start_time)
                                                                                                  - costs_btw
                                                                                                , - costs_btw) \
                                                              if travel_allowed  else self.LARGE_NUMBER

        return value_vec

    def __freight_hedge_action(self) -> Tuple[np.array, np.array]:
        """ Find optimum freight hedge, solve the linear program.

        @returns: a list of hedge transport schedule, and a list of values associated w/ that transport
        """

        em, ev = self.__EMV_mat()  # equality matrix condition em * x = ev

        lm_mat = self.__lm_mat()
        value_vec = self._value()
        result = linprog( value_vec
                        , A_ub = lm_mat  # inequality condition A_ub * x <= b_ub
                        , b_ub = np.zeros(lm_mat.shape[0])  # zeros the shape of LMMat
                        , A_eq = em
                        , b_eq = ev )

        if result.success:
            return result.x, value_vec

        raise FreightException(result.message)


    def _freight_hedge(self) -> Tuple[np.array, np.array]:
        """ Find optimum freight hedge, solve the linear program.

        @returns: tuple of optimal transport schedule, value of each of the transport options.
        """

        if self.__recompute_hedge or self.__freight_hedge is None or self.__transport_value is None:
            self.__freight_hedge, self.__transport_value = self.__freight_hedge_action()
            self.__recompute_hedge = False  # no need to recompute
            return self.__freight_hedge, self.__transport_value

        return self.__freight_hedge, self.__transport_value

    def _hedge_locations(self, ignore_small_nbs = .0001) -> Dict[datetime.date, Dict[str, float]]:
        """ Represents the hedge locations of the freight problem.

        @param ignore_small_nbs: ignore hedges below this number
        @returns: two level dictionary where the first level is the time period, and the second level is the
                  location of the tanker. The key is the number of tankers.
                  e.g. {(2015, 1, 1): { 'AMS' : 2, 'MIA': 3},
                        (2015, 1, 15): {'AMS': 3, 'MIA': 2} }
        """

        freight_hedge, _ = self._freight_hedge()

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

    def _hedge_movements_cond_uncond( self
                                    , cond_uncond      : str
                                    , ignore_small_nbs : float = 0.001) -> Dict[datetime.date, Dict[str, Tuple[str, datetime.date, float, float]]]:
        """ Conditional or unconditional representation of hedge movements.

        @param cond_uncond: 'cond' if conditional movments, else 'uncond'
        @param ignore_small_nbs: number below which the hedge positions are ignored.
        @returns: double dictionary where the first rank is the date of the start of the movement,
                  the second key is location of start of the transport
                  the list of tuples is where the distribution goes to e.g.
                        (AMS, (2015, 6, 15), 3, 15.) means 3 tankers are going to AMS for delivery there at 6/15, total value of this is 15.
        """

        hedge, trans_value = self._freight_hedge()
        cond_uncond_var    = self._X if cond_uncond == 'cond' else self._Y

        hm_cond = {}

        for time_period_start in range(self._nb_time_periods):
            start_time = self._time_grid[time_period_start]
            hm_cond[start_time] = {}

            for location_start in self._locations: # for i in range(self._nb_locations)}  # location was i = self._locations.index(location)
                location_start_nb = self._locations.index(location_start)
                hm_cond[start_time][location_start] = []

                for location_end_nb in range(self._nb_locations):
                    for time_period_end in range(time_period_start + 1, self._nb_time_periods):
                        variable_loc    = cond_uncond_var(location_start_nb, location_end_nb, time_period_start, time_period_end)
                        hedged_value    = hedge[variable_loc]
                        transport_value = - trans_value[variable_loc]  # negative since the value in the opt. program is minimized.

                        if hedged_value > ignore_small_nbs:  # only include if the numbers are sufficiently large
                            location_end = self._nbs_to_locations[location_end_nb]
                            end_time     = self._time_grid[time_period_end]
                            hm_cond[start_time][location_start].append( ( location_end
                                                                        , end_time
                                                                        , hedged_value
                                                                        , hedged_value * transport_value) )

                if hm_cond[start_time][location_start] == []:
                    hm_cond[start_time].pop(location_start)  # remove location

            if hm_cond[start_time] == {}:  # remove time period
                hm_cond.pop(start_time)

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

    def represent_hedge(self, ignore_small_nbs = .0001 ) -> Dict:
        """ Represents the hedge obtained from optimization.

        @param ignore_small_nbs: ignore all hedges below a certain threshold, e.g. 1e-4
        @returns:
        """

        transport_sched, transport_value = self._freight_hedge()

        return { 'portfolio_value' : - np.sum(transport_value * transport_sched)  # self._value is negative, cause linprog is minimized
               , 'locations'       : self._hedge_locations(ignore_small_nbs)
               , 'movements_cond'  : self._hedge_movements_cond_uncond('cond', ignore_small_nbs)
               , 'movements_uncond': self._hedge_movements_cond_uncond('uncond', ignore_small_nbs)
               , }

    def __merge_dest_lists( dest_list_1 : List[Tuple[str, datetime.date, float, float]]
                          , dest_list_2 : List[Tuple[str, datetime.date, float, float]]) -> List[Tuple[str, datetime.date, float, float]]:
        """ Merges two destination lists.

        @param dest_list_1: destination list 1 of the form [('AMS', datetime.date(2015, 4, 1), 1, 10.),...]
        @param dest_list_2: destination list 2 of the form [('AMS', datetime.date(2015, 4, 1), 1, 10.),...]
        @returns: destination lists where things are merged.
        """

        result = copy.deepcopy(dest_list_1)

        dest_dates_1 =  [(dest_1, dest_date_1) for dest_1, dest_date_1, nb_tankers_1 in dest_list_1]

        for destination_2, dest_date_2, nb_tankers_2, nb_value_2 in dest_list_2:
            if (destination_2, dest_date_2) in dest_dates_1:
                matching_idx = dest_dates_1.index((destination_2, dest_date_2))
                _, _, nb_tankers_1, nb_value_1 = result[matching_idx]
                result[matching_idx] = (destination_2, dest_date_2, nb_tankers_2 + nb_tankers_1, nb_value_1 + nb_value_2)

            else:
                result.append((destination_2, dest_date_2, nb_tankers_2, nb_value_2))

        return result


    def show_dynamics(self, ignore_small_nbs : float = 0.0001) -> Dict:
        """ Extract the tanker routes from the hedges.

        @param ignore_small_nbs: ignore all hedges below a certain threshold, e.g. 1e-4
        @returns:
        """

        rh          = self.represent_hedge()  # rv ... represent value
        move_cond   = rh['movements_cond'  ]
        move_uncond = rh['movements_uncond']

        cond_dates   = sorted(list(move_cond.keys()))
        uncond_dates = sorted(list(move_uncond.keys()))
        all_dates    = sorted(set(cond_dates).union(set(uncond_dates)))
        overlapping_dates = sorted(set(cond_dates).intersection(set(uncond_dates)))

        ship_schedule = {}

        for start_date in all_dates:
            if start_date in overlapping_dates:  # add from both.
                ship_schedule[start_date] = move_cond[start_date]  # done w/ move_cond
                curr_ship_schedule = ship_schedule[start_date]  #  {'NYC': [('AMS', dest_date, nb_tankers),...] }

                # (destination, dest_date, nb_tankers)
                for origin, dest_list in move_uncond[start_date].items():
                    if origin not in curr_ship_schedule:
                        curr_ship_schedule[origin] = dest_list
                    else:  # origin is in ship_schedule, merge the two lists lists
                        curr_ship_schedule[origin] = self.__merge_dest_lists(curr_ship_schedule[origin], dest_list)

            elif start_date in move_cond and start_date not in move_uncond:
                ship_schedule[start_date] = move_cond[start_date]

            elif start_date in move_uncond and start_date not in move_cond:
                ship_schedule[start_date] = move_uncond[start_date]

        return { 'pv'       : rh['portfolio_value']
               , 'locations': rh['locations']
               , 'schedule' : ship_schedule }

    def show_dynamics_and_locations(self, ignore_small_nbs : float = 0.0001) -> None:
        sd       = self.show_dynamics()
        locs     = sd['locations']
        schedule = sd['schedule']

        locs_dates     = set(locs.keys())
        schedule_dates = set(schedule.keys())

        for start_date in sorted(schedule_dates.union(locs_dates)):
            print('DATE: {0}'.format(start_date))
            if start_date in locs_dates:
                print('LOCATIONS')
                self.pretty_dict(locs[start_date])
            if start_date in schedule_dates:
                print('SCHEDULE')
                self.pretty_dict(schedule[start_date])
            print('\n')


class FreightAdded(Freight):

    def __init__(self
                 , mkt_date          : datetime.date
                 , fwd_curves        : Callable
                 , vol_curves        : Callable
                 , corr_matrix       : Dict[Tuple[str, str], float]
                 , travel_times      : Dict[Tuple[str, str], int]
                 , cost_matrix       : Dict[Tuple[str, str], float]
                 , initial_locations : Dict[str, Tuple[datetime.date, int]]
                 , final_date        : datetime.date
                 , dcf               : float = 365.25 ):

        # compute time grid from initial_locations and final date
        all_frequencies = {travel_days for route, travel_days in travel_times.items()}
        all_dates = set()  # set of all dates for the time_grid
        for frequency in all_frequencies:
            initial_start_dates = {start_date for _, (start_date, _) in initial_locations.items()}
            initial_start_dates.add(mkt_date)
            for initial_start_date in initial_start_dates:
                curr_date = initial_start_date
                while curr_date < final_date:
                    all_dates.add(curr_date)
                    curr_date += datetime.timedelta(days=frequency)

        super().__init__( mkt_date
                          , fwd_curves
                          , vol_curves
                          , corr_matrix
                          , travel_times
                          , cost_matrix
                          , initial_locations
                          , sorted(list(all_dates))
                          , dcf = dcf )
