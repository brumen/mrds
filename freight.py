# Freight model implementation
#

import datetime, numpy as np, logging
from scipy.optimize import linprog

from ds import DF
from pricers.pricers import spread_option_kirk


class FreightException(Exception):
    pass


LARGE_NUMBER = 1000000.  # large number to prohibit travel between certain directions & times.
logger = logging.getLogger(__name__)


class Freight(object):
    """
    Freight class:
      N ... initial distribution of the tanker fleet
    """

    def __init__( self
                , mktDate          : datetime.date
                , fwdCurveFct  # function
                , volCurveFct  # function
                , corrMatrix       : dict
                , travelMatrix     : dict
                , costMatrix       : dict
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
        :param costMatrix: same as travelMatrix, but refers to costs between cities.
        :param timeGrid: time grid for the problem, meaning that the  list[datetime.date]
        :param dcf: day count factor, used for discounting and option evaluation.
        """

        self.mktDate      = mktDate
        self.fwdCurveFct  = fwdCurveFct
        self.volCurveFct  = volCurveFct
        self._corrMatrix  = corrMatrix
        self._travelMatrix = travelMatrix  # number of periods between different locations
        self._costMatrix   = costMatrix    # same as travel matrix, costs between locations
        self._timeGrid    = timeGrid       # grid used to compute the value of the freight portfolio.
        self._dcf         = dcf            # day count factor
        self._initialLocations = initialLocations  # initial locations of the portfolio

        # simple derived variables
        self._locations       = initialLocations.keys()  # locations considered are given in initialLocations
        self._nbLocations     = len(self._locations)     # number of different locations
        self.__nbsToLocations = {idx: loc for (idx, loc) in enumerate(self._locations)}
        self.__nbsToTimeGrid  = {idx: timeStep for (idx, timeStep) in enumerate(self._timeGrid) }
        self.__nbTimePeriods  = len(self._timeGrid)                # length of grid = number of time periods + 1

        # cached values, used as properties
        self.__valueVec   = None  # vector of all individual values
        self.__LM         = None
        self.__EM         = None
        self.__EV         = None
        self.__lowerBound = None

    def fwdVolCurves( self
                    , location : str
                    , futureDate : datetime.date
                    , fwdVolInd = 'fwd' ) -> float:
        """
        Gets the forward curves for market date for the times requested in timeList

        :param location: location, string
        :param futureDate: time for which the forward curve is requested
        :param fwdVolIndict: indicating whether 'fwd' or 'vol' is computed (string)
        :returns: array of forwards or vols for that timeList and location (returns vector)
        """

        return (self.fwdCurveFct if fwdVolInd == 'fwd' else self.volCurveFct)(self.mktDate, location, futureDate)

    @property
    def lowerBound(self):
        """
        Lower bound of the individual variables, which is 0, basically a vector of zeros.

        """

        if self.__lowerBound is not None:
            return self.__lowerBound

        self.__lowerBound = np.zeros(self._nbLPVariables)

        return self.__lowerBound

    def spreadOption( self
                    , city1 : str
                    , city2 : str
                    , t1    : datetime.date
                    , t2    : datetime.date ) -> float:
        """
        Spread option value between city1, city2 and times t1, t2, t1<t2.
        """

        return spread_option_kirk( self.fwdVolCurves(city1, t1)
                                 , self.fwdVolCurves(city2, t2)
                                 , self._costMatrix[(city1, city2) if (city1, city2) in self._costMatrix else (city2, city1)] if city1 != city2 else 0.
                                 , self.fwdVolCurves(city1, t1, fwdVolInd='vol')
                                 , self.fwdVolCurves(city2, t2, fwdVolInd='vol')
                                 , self._corrMatrix[(city1, city2) if (city1, city2) in self._corrMatrix else (city2, city1)]
                                 , (t2 - t1).days / self._dcf
                                 , DF(self.mktDate, t2))

    def _X(self, i : int, j : int, t : int, u :int) -> int:
        """
        Conditional transport variable.
        Location of the variable x_(i,j,t,u) in the matrix, t<u
        Index corresponding to shipping from city i to city j between time t and u conditional.

        """

        return i + j * self._nbLocations + self._nbLocations**2 * (self.__nbTimePeriods - 1 - u + (self.__nbTimePeriods * t - t * (t+1)//2))

    def _Y(self, i : int, j: int, t : int, u: int) -> int:
        """

        Unconditional transport variable.
        """

        # first line is the number of variables X
        return self._nbLocations**2 * self.__nbTimePeriods* (self.__nbTimePeriods - 1)//2 \
               + self._X(i, j, t, u)

    def _N(self, i, t) -> int:
        """
        Number of tankers in city i at time t. Location of the variable n_(i,t) in the vector of all variables.
        """

        return self._nbLocations**2 * self.__nbTimePeriods* (self.__nbTimePeriods - 1) \
               + i + t * self._nbLocations

    @property
    def _nbLPVariables(self) -> int:
        """
        Number of variables for the Linear problem. (X, Y, Z, N)
        """

        return self._nbLocations**2 * self.__nbTimePeriods* (self.__nbTimePeriods - 1) \
               + self.__nbTimePeriods * self._nbLocations

    def _travelAllowed(self, i:int, j:int, u:int, t:int) -> bool:
        """
        Is travel between i and j allowed between times u & t: t> u
        """

        if i == j:  # that is always allowed
            return True

        city_i, city_j = self.__nbsToLocations[i], self.__nbsToLocations[j]

        return t - u >= self._travelMatrix[(city_i, city_j) if (city_i, city_j) in self._travelMatrix else (city_j, city_i)]

    @property
    def _LMMat(self) -> np.array:
        """
        Constructs the inequality matrix self.__LM, i.e. for conditions:
        self.__LM * x <= 0. (0. is a vector)
        """

        if self.__LM is not None:
            return self.__LM

        consMat = []  # constraint matrix

        # constraint n_{i,t} >= sum_{j,u} X(i,j,t,u) + Y(i,j,t,u)
        for t in range(self.__nbTimePeriods):
            for i in range(self._nbLocations):
                consVec = np.zeros(self._nbLPVariables)
                consVec[self._N(i, t)] = -1.
                for j in range(self._nbLocations):
                    for u in range(t+1, self.__nbTimePeriods):
                        consVec[self._X(i, j, t, u)] = 1.
                        consVec[self._Y(i, j, t, u)] = 1.
                consMat.append(consVec)

        self.__LM = np.array(consMat)

        return self.__LM

    @property
    def _EMVMat(self) -> (np.array, np.array):
        """
        Sets the equality matrix (EM) and equality vector (EV), for constraints:
        EM * x = EV, EM is a matrix, EV is a vector.
        """

        if (self.__EM is not None) and (self.__EV is not None):
            return self.__EM, self.__EV

        __equalityMatrix = []
        __equalityVector = []

        # initial setting of N: N(i,0) = initialLocation(i)
        for i in range(self._nbLocations):
            consVec = np.zeros(self._nbLPVariables)
            consVec[self._N(i, 0)] = 1.
            __equalityVector.append(self._initialLocations[self.__nbsToLocations[i]])
            __equalityMatrix.append(consVec)

        # sum_i n_i,t = K 
        for t in range(1, self.__nbTimePeriods):  # t=0 already given above
            consVec = np.zeros(self._nbLPVariables)
            for i in range(self._nbLocations):
                consVec[self._N(i, t)] = 1.
            __equalityVector.append(sum(self._initialLocations.values()))  # number of tankers, ships
            __equalityMatrix.append(consVec)

        # constraint n_i,t = n_i,t-1 + sum_{j, u<t} (X(j, i, u, t) + Y(j,i,t,u)) - sum _{u>t-1, j} (X(i,j,t-1, u) + Y(i,j,t-1,u))
        for t in range(1, self.__nbTimePeriods):
            for i in range(self._nbLocations):
                consVec = np.zeros(self._nbLPVariables)  # consVec is constraints vector
                consVec[self._N(i, t)]     =  1.
                consVec[self._N(i, t - 1)] = -1.
                for j in range(self._nbLocations):
                    for u in range(t):
                        consVec[self._X(j, i, u, t)] = -1.
                        consVec[self._Y(j, i, u, t)] = -1.
                    for u in range(t, self.__nbTimePeriods):
                        consVec[self._X(i, j, t - 1, u)] = 1.
                        consVec[self._Y(i, j, t - 1, u)] = 1.
                __equalityMatrix.append(consVec)
                __equalityVector.append(0.)

        self.__EM = np.array(__equalityMatrix)
        self.__EV = np.array(__equalityVector)

        return self.__EM, self.__EV

    @property
    def _valueVec(self):
        """
        Setting the valuation vector - this is set for _MINIMIZING (COST) FUNCTION, therefore the values are negative.

        """

        if self.__valueVec is not None:
            return self.__valueVec

        self.__valueVec = np.zeros(self._nbLPVariables)

        for t in range(self.__nbTimePeriods):  # time period
            for i in range(self._nbLocations):  # cities
                for j in range(self._nbLocations):
                    for u in range(t+1, self.__nbTimePeriods):
                        self.__valueVec[self._X(i, j, t, u)] = - self.spreadOption( self.__nbsToLocations[i]
                                                                                  , self.__nbsToLocations[j]
                                                                                  , self.__nbsToTimeGrid[t]
                                                                                  , self.__nbsToTimeGrid[u]) if self._travelAllowed(i,j,t,u) else LARGE_NUMBER
                        self.__valueVec[self._Y(i,j,t,u)] = -(self.fwdVolCurves(self.__nbsToLocations[j], self.__nbsToTimeGrid[u])
                                                              - self.fwdVolCurves(self.__nbsToLocations[i], self.__nbsToTimeGrid[t])) \
                                                              if self._travelAllowed(i,j,t,u) else LARGE_NUMBER

        return self.__valueVec

    def freightHedge(self):
        """
        Find optimum freight hedge, solve the linear program.

        """

        EM, EV = self._EMVMat  # equality matrix condition EM * x = EV

        result = linprog( self._valueVec
                        , A_ub   = self._LMMat  # inequality condition A_ub * x <= b_ub
                        , b_ub   = np.zeros(self._LMMat.shape[0])  # zeros the shape of LMMat
                        , A_eq   = EM
                        , b_eq   = EV )
                        # , bounds = list(zip(self.lowerBound, [None] * len(self.lowerBound) ) ) )  # this bounds are by default.

        if result.success:
            return result.x  # actual result
        else:
            raise FreightException(result.message)

    def representHedge(self):
        """
        Represents the hedge obtained from optimization.

        """

        result = self.freightHedge()

        tankerLocations = { self._timeGrid[t]: {self.__nbsToLocations[i]: result[self._N(i,t)] for i in range(self._nbLocations)}
                            for t in range(self.__nbTimePeriods) }

        tankerMovementsConditional = { self._timeGrid[t]: {self.__nbsToLocations[i]: [(self.__nbsToLocations[j], self._timeGrid[u], result[self._X(i, j, t, u)])
                                                                                      for j in range(self._nbLocations) for u in range(t + 1, self.__nbTimePeriods)]
                                                           for i in range(self._nbLocations)}
                                       for t in range(self.__nbTimePeriods)}

        tankerMovementsUnConditional = { self._timeGrid[t]: {self.__nbsToLocations[i]: [(self.__nbsToLocations[j], self._timeGrid[u], result[self._Y(i, j, t, u)])
                                                                                        for j in range(self._nbLocations) for u in range(t + 1, self.__nbTimePeriods)]
                                                             for i in range(self._nbLocations)}
                                         for t in range(self.__nbTimePeriods)}

        return { 'portfolioValue': - np.sum(np.array(self._valueVec) * np.array(result))  # self.valueVec is negative, cuase linprog is minimized
               , 'locations'     : tankerLocations
               , 'movements'     : (tankerMovementsConditional, tankerMovementsUnConditional) }
