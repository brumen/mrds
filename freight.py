# Freight model implementation
#

import datetime, numpy as np
from scipy.optimize import linprog

from ds import DF
from pricers.pricers import spread_option_kirk_zero_strike


class Freight(object):
    """
    Freight class:
      N ... initial distribution of the fleet ... IN WHICH FORMAT SHOULD THIS BE?
      K ... number of tankers ... CHECK CHECK CHECK
      I ... number of cities
      T ... number of time periods
    """

    def __init__( self
                , mktDate          : datetime.date
                , fwdCurveFct  # function
                , volCurveFct  # function
                , corrMatrix       : dict
                , travelMatrix     : dict
                , initialLocations : dict
                , timeGrid
                , dcf = 365.25):
        """
        :param mktDate: market date
        :param locations: locations between which freight can be transported, list[str]
        :param fwdCurveFct: fucntion of (location, mktDate, future date), returns forward rate for that point.
        :param initialLocations: dictionary of how many ships are in a particular location.
                                 {location: nb_ships }
        :param corrMatrix: correlation between individual locations, dictionary where keys are
                           city pairs (city_1, city_2) and values are correlations between cities.
        :param travelMatrix: the amount of time it takes between different locations, a dictionary
                             where keys are location pairs (loc_1, loc_2) and values are time as fractions of
                             a year (i.e. 1. means 1 year).
        :param timeGrid: time grid for the problem, meaning that the  list[datetime.date]
        :param dcf: day count factor, used for discounting and option evaluation.
        """

        self.mktDate      = mktDate
        self.fwdCurveFct  = fwdCurveFct
        self.volCurveFct  = volCurveFct
        self._corrMatrix  = corrMatrix
        self._travelMatrix = travelMatrix  # number of periods between different locations
        self._timeGrid    = timeGrid       # grid used to compute the value of the freight portfolio.
        self._dcf         = dcf            # day count factor
        self._initialLocations = initialLocations  # initial locations of the portfolio

        # simple derived variables
        self._locations       = initialLocations.keys()  # locations considered are given in initialLocations
        self._nbLocations     = len(self._locations)     # number of different locations
        self.__nbsToLocations = {idx: loc for (idx, loc) in enumerate(self._locations)}
        self.__nbsToTimeGrid  = {idx: timeStep for (idx, timeStep) in enumerate(self._timeGrid) }
        self.__nbTimePeriods  = len(self._timeGrid)                # number of time periods

        # cached values, used as properties
        self.__valueVec = None  # vector of all individual values
        self.__LM   = None
        self.__EM   = None
        self.__EV   = None
        self.__lowerBound = None

    def fwdVolCurves( self
                    , location : str
                    , timeList : list
                    , fwdVolInd = 'fwd' ) -> np.array:
        """
        Gets the forward curves for market date for the times requested in timeList

        :param location: location, string
        :param timeList: list of times for which the forward curve is requested, list[datetime.date]
        :param fwdVolIndict: indicating whether 'fwd' or 'vol' is computed (string)
        :returns: array of forwards or vols for that timeList and location (returns vector)
        """

        fwdVolFctUsed = self.fwdCurveFct if fwdVolInd == 'fwd' else self.volCurveFct

        return np.array([fwdVolFctUsed(self.mktDate, location, t)
                         for t in timeList]) if type(timeList) is list else fwdVolFctUsed(self.mktDate, location, timeList)

    @property
    def lowerBound(self):
        """
        Lower bound of the individual variables, which is 0, basically a vector of zeros.

        """

        if self.__lowerBound is not None:
            return self.__lowerBound

        self.__lowerBound = np.zeros(self._nbLocations**2 * self.__nbTimePeriods * (self.__nbTimePeriods - 1)//2 +
                                     3 * self._nbLocations * self.__nbTimePeriods )

        return self.__lowerBound

    def V( self
         , city1 : str
         , city2 : str
         , t1    : datetime.date
         , t2    : datetime.date ):
        """
        construction of spread option matrices
        between cities city1, city2, and times t1, t2
        between times t_1 and t_2 and between cities c_1 and c_2.
        V[t_1][t_2] [c_1, c_2], t_1 < t_2

        """

        # reverse cities if necessary
        c1_used, c2_used = (city1, city2) if (city1, city2) in self._corrMatrix else (city2, city1)

        return spread_option_kirk_zero_strike( self.fwdVolCurves(city1, t1)
                                             , self.fwdVolCurves(city2, t2)
                                             , self.fwdVolCurves(city1, t1, fwdVolInd='vol')
                                             , self.fwdVolCurves(city2, t2, fwdVolInd='vol')
                                             , self._corrMatrix[(c1_used, c2_used)]
                                             , (t2 - t1).days / self._dcf
                                             , DF(self.mktDate, t1))

    def X(self, i : int, j : int , t : int, u :int ):
        """
        Location of the variable x_(i,j,t,u) in the matrix, t<u
        Index corresponding to shipping from city i to city j between time t and u. This is a BENEFIT.

        """

        return i + j * self._nbLocations + self._nbLocations**2 * (self.__nbTimePeriods - 1 - u + (self.__nbTimePeriods * t - t * (t+1)//2))

    def Y(self, i : int, t : int):
        """
        Injection at location i at time t. Location of the variable y_(i,t) in the matrix. This is a BENEFIT.
        """

        return self._nbLocations**2 * self.__nbTimePeriods * (self.__nbTimePeriods - 1)//2 + i + t * self._nbLocations

    def Z(self, i, t):
        """
        Withdrawal at i, t. Location of the variable z_(i,t) in the matrix. This is a COST.

        """

        return self._nbLocations**2 * self.__nbTimePeriods * (self.__nbTimePeriods - 1)//2 + self._nbLocations * self.__nbTimePeriods + i + t * self._nbLocations

    def N(self, i, t):
        """
        Initial value of the tankers. Location of the variable n_(i,t) in the matrix
        """

        return self._nbLocations**2 * self.__nbTimePeriods * (self.__nbTimePeriods - 1)//2 + 2*self._nbLocations*self.__nbTimePeriods + i + t * self._nbLocations

    @property
    def LMMat(self):
        """
        sets the inequality matrix LM (lower matrix???)

        """

        if self.__LM is not None:
            return self.__LM

        self.__LM = np.zeros((self._nbLocations**2 * self.__nbTimePeriods * (self.__nbTimePeriods-1)//2 + 4 * self._nbLocations * self.__nbTimePeriods,
                              self._nbLocations**2 * self.__nbTimePeriods * (self.__nbTimePeriods-1)//2 + 3 * self._nbLocations * self.__nbTimePeriods))

        # constraint n(i,t) - y(i,t) + z(i,t) > 0
        row_ineq_idx = 0  # initial row idx for inequalities
        for t in range(self.__nbTimePeriods):
            for i in range(self._nbLocations):
                self.__LM[row_ineq_idx, self.N(i, t)] = -1.
                self.__LM[row_ineq_idx, self.Z(i, t)] = -1.
                self.__LM[row_ineq_idx, self.Y(i, t)] = 1.
                row_ineq_idx += 1

        # constraint x_i,j,t,u >= 0
        for i in range(self._nbLocations):
            for j in range(self._nbLocations):
                for t in range(self.__nbTimePeriods-1):  # -1 is important here
                    for u in range(t+1, self.__nbTimePeriods):
                        self.__LM[row_ineq_idx, self.X(i, j, t, u)] = -1
                        row_ineq_idx += 1

        # constraints n_i,t and y_i,t and z_i,t >= 0
        for t in range(self.__nbTimePeriods):
            for i in range(self._nbLocations):
                self.__LM[row_ineq_idx, self.Y(i, t)] = -1.
                row_ineq_idx += 1
                self.__LM[row_ineq_idx, self.Z(i, t)] = -1.
                row_ineq_idx += 1
                self.__LM[row_ineq_idx, self.N(i, t)] = -1.
                row_ineq_idx += 1

        return self.__LM

    def _travelAllowed(self, i:int, j:int, u:int, t:int):
        """
        Is travel between i and j allowed between times u & t:
        t> u
        """

        if i == j:  # that is always allowed
            return True

        city_i, city_j = self.__nbsToLocations[i], self.__nbsToLocations[j]
        city_i_used, city_j_used = (city_i, city_j) if (city_i, city_j) in self._travelMatrix else (city_j, city_i)

        return t - u >= self._travelMatrix[(city_i_used, city_j_used)]

    @property
    def EMVMat(self):
        """
        Sets the equality matrix (EM) and equality vector (EV)

        """

        if (self.__EM is not None) and (self.__EV is not None):
            return self.__EM, self.__EV

        # equality matrix
        self.__EM = np.zeros((self._nbLocations * self.__nbTimePeriods + self.__nbTimePeriods - 1,
                              self._nbLocations**2 * self.__nbTimePeriods * (self.__nbTimePeriods-1)//2 + 3 * self._nbLocations * self.__nbTimePeriods))

        # equality vector
        self.__EV = np.zeros(self._nbLocations * self.__nbTimePeriods + self.__nbTimePeriods - 1)

        row_eq_idx = 0  # row indx. for equalities

        # initial setting of N: N(i,0) = initialLocation(i)
        for i in range(self._nbLocations):
            self.__EM[row_eq_idx, self.N(i, 0)] = 1.
            self.__EV[row_eq_idx] = self._initialLocations[self.__nbsToLocations[i]]
            row_eq_idx += 1

        # sum_i n_i,t = K 
        for t in range(1, self.__nbTimePeriods):  # t=0 already given above
            for i in range(self._nbLocations):
                self.__EM[row_eq_idx, self.N(i, t)] = 1.
            self.__EV[row_eq_idx] = sum(self._initialLocations.values())  # number of tankers, ships
            row_eq_idx += 1

        # constraint n_i,t = n_i,t-1 + sum_{j, u<t} X(j, i, u, t) - sum_{j, u>t) X_{i, j, t, u} - y(i,t) + z(i,t)
        for t in range(1, self.__nbTimePeriods):
            for i in range(self._nbLocations):
                self.__EM[row_eq_idx, self.N(i, t)]   = 1.
                self.__EM[row_eq_idx, self.N(i, t-1)] = -1.
                self.__EM[row_eq_idx, self.Y(i,t)]    = -1.
                self.__EM[row_eq_idx, self.Z(i, t)] = 1.
                for j in range(self._nbLocations):
                    for u in range(t):
                        if self._travelAllowed(i, j, u, t):
                            self.__EM[row_eq_idx, self.X(j, i, u, t)] = -1.
                    for u in range(t+1, self.__nbTimePeriods):
                        if self._travelAllowed(i, j, t, u):  # only if travel between cities is allowed
                            self.__EM[row_eq_idx, self.X(i, j, t, u)] = 1.
                row_eq_idx += 1

        return self.__EM, self.__EV

    @property
    def valueVec(self):
        """
        Setting the valuation vector - this is set for _MINIMIZING (COST) FUNCTION, THEREFORE THE SIGNS ARE INVERSED.

        """

        if self.__valueVec is not None:
            return self.__valueVec

        self.__valueVec = np.zeros(self._nbLocations ** 2 * self.__nbTimePeriods * (self.__nbTimePeriods - 1) // 2 + 3 * self._nbLocations * self.__nbTimePeriods)

        fwdCurves = [self.fwdVolCurves(location, self._timeGrid) for location in self._locations]

        for t in range(self.__nbTimePeriods):  # time period
            for i in range(self._nbLocations):  # cities
                self.__valueVec[self.Z(i, t)] =   fwdCurves[i][t]  # this is a cost
                self.__valueVec[self.Y(i, t)] = - fwdCurves[i][t]  # this is a benefit
                for j in range(self._nbLocations):
                    for u in range(t+1, self.__nbTimePeriods):
                        self.__valueVec[self.X(i, j, t, u)] = - self.V( self.__nbsToLocations[i]
                                                                      , self.__nbsToLocations[j]
                                                                      , self.__nbsToTimeGrid[t]
                                                                      , self.__nbsToTimeGrid[u] )  # this are benefits.

        return self.__valueVec

    def freightHedge(self):
        """
        Find optimum freight hedge.

        """

        EM, EV = self.EMVMat

        result =linprog( self.valueVec
                       , A_ub   = self.LMMat
                       , A_eq   = EM
                       , b_ub   = np.zeros((self._nbLocations**2 * self.__nbTimePeriods * (self.__nbTimePeriods-1)//2 + 4 * self._nbLocations * self.__nbTimePeriods, 1))
                       , b_eq   = EV
                       , bounds = list(zip(self.lowerBound, [None] * len(self.lowerBound) ) ) )

        return result.x  # actual result

    def representHedge(self):
        """
        Represents the hedge obtained from optimization.

        """

        result = self.freightHedge()
        value  = - np.sum(np.array(self.valueVec) * np.array(result))  # self.valueVec is negative, cuase linprog is minimized

        print('Portfolio value: {0}'.format(value))

        # TODO: THIS IS BRUTE FORCE, COULD DEFINITELY BE IMPROVED.
        for t in range(self.__nbTimePeriods):  # time period
            for i in range(self._nbLocations):  # cities
                if result[self.Z(i,t)] != 0.:
                    print("Selling {2} from city {0} on {1}".format(self.__nbsToLocations[i], self._timeGrid[t], result[self.Z(i,t)]))
                if result[self.Y(i,t)] != 0:
                    print("Buying {2} into city {0} on {1}".format(self.__nbsToLocations[i], self._timeGrid[t], result[self.Y(i,t)]))
                for j in range(self._nbLocations):
                    for u in range(t + 1, self.__nbTimePeriods):  # TODO: CHECK IF THIS IS t+1 or NOT
                        if result[self.X(i, j, t, u)] != 0.:
                            print("Buying from {0}/Selling to {1} between {2} - {3} : {4}".format( self.__nbsToLocations[i]
                                                                                                 , self.__nbsToLocations[j]
                                                                                                 , self._timeGrid[t]
                                                                                                 , self._timeGrid[u]
                                                                                                 , result[self.X(i,j,t,u)]))
