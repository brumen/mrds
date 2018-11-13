# Freight model implementation
#

import datetime, numpy as np
from scipy.optimize import linprog

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
                , mktDate     : datetime.date
                , fwdCurveFct : function
                , volCurveFct : function
                , corrMatrix  : dict
                , travelMatrix : dict
                , initialLocations : dict
                , timeGrid ):
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
        """

        self.mktDate      = mktDate
        self._locations   = initialLocations.keys()  # locations considered are given in initialLocations
        self._nbLocations = len(self._locations)
        self.nbs_to_locs  = {idx: loc for (idx, loc) in enumerate(self._locations)}
        self.fwdCurveFct  = fwdCurveFct
        self.volCurveFct  = volCurveFct
        self._corrMatrix   = corrMatrix
        self.travelMatrix = travelMatrix            # number of periods between different locations
        self._timeGrid     = timeGrid

        self.__K = np.sum(initialLocations.values())  # number of tankers
        self.__T = len(self._timeGrid)                # number of time periods
        self.__I = len(self._locations)               # number of cities

        # cached values
        self.__Vmat = None
        self.__LM   = None
        self.__EM   = None
        self.__EV   = None
        self.__lowerBound = None

    def fwdVolCurves( self
                    , location : str
                    , timeList : list
                    , fwdVolInd : str ):
        """
        Gets the forward curves for market date for the times requested in timeList

        :param location: location, string
        :param timeList: list of times for which the forward curve is requested, list[datetime.date]
        :param fwdVolIndict: indicating whether 'fwd' or 'vol' is computed (string)
        """

        return np.array([(self.fwdCurveFct if fwdVolInd == 'fwd' else self.volCurveFct)(self.mktDate, location, t)
                         for t in timeList])

    @property
    def lowerBound(self):
        """
        Lower bound of the individual variables.
        """

        if self.__lowerBound is not None:
            return self.__lowerBound

        self.__lowerBound = np.zeros(self.__I**2 * self.__T * (self.__T - 1)//2 + 3 * self.__I * self.__T )

        return self.__lowerBound

    @property
    def V(self):
        """
        construction of spread option matrices
        between times t_1 and t_2 and between cities c_1 and c_2.
        V[t_1][t_2] [c_1, c_2], t_1 < t_2

        """

        if self.__Vmat:
            return self.__Vmat

        self.__Vmat = [[0] * self.__T] * self.__T  # by time periods

        fwdCurves = [[self.fwdVolCurves(location, t) for t in timeList] for location in self._locations]
        volCurves = [[self.fwdVolCurves(location, t, fwdVolInd='vol') for t in timeList] for location in self._locations]

        # construction of spread option
        for t_1 in range(self.__T):
            for t_2 in range((t_1+1), self.__T):
                self.__Vmat[t_1][t_2] = np.zeros((self.nb_locs, self.nb_locs))
                for c_1_nb, c_1 in enumerate(self.locs):  # city 1
                    for c_2_nb, c_2 in enumerate(self.locs):  # city 2
                        if (c_1, c_2) in self.corr_mtx:
                            c1_used, c2_used = c_1, c_2
                        else:
                            c1_used, c2_used = c_2, c_1
                        # spread_option_kirk (F_1, F_2, K, sigma_1, sigma_2, rho, T, DF)
                        self.__Vmat[t_1][t_2][c_1_nb, c_2_nb] = spread_option_kirk_zero_strike( fwdCurves[c_1][t_1]
                                                                                              , fwdCurves[c_2][t_2]
                                                                                              , volCurves[c_1][t_1]
                                                                                              , volCurves[c_2][t_2]
                                                                                              , self.corr_mtx[(c1_used, c2_used)]
                                                                                              , t_2 - t_1
                                                                                              , 0.99)  # DISCOUNT FACTOR IS WRONG

        return self.__Vmat

    def X(self, i : int, j : int , t : int, u :int ):
        """
        Index corresponding to shipping from city i to city j between time t and u.

        """

        return i + j * self.__I + self.__I**2 * (self.__T - 1 - u + (self.__T * t - t * (t+1)//2))

    def Y(self, i, t):
        """

        """

        return self.__I**2 * self.__T * (self.__T - 1)//2 + i + t * self.__I

    def Z(self, i, t):
        """


        """
        return self.__I**2 * self.__T * (self.__T - 1)//2 + self.__I * self.__T + \
            i + t * self.__I

    def N(self, i, t):
        """

        """
        return self.__I**2 * self.__T * (self.__T - 1)//2 + 2*self.__I*self.__T + \
            i + t * self.__I

    @property
    def LMMat(self):
        """
        sets the inequality matrix LM (lower matrix???)

        """

        if self.__LM is not None:
            return self.__LM

        self.__LM = np.zeros((self.__I**2 * self.__T * (self.__T-1)//2 + 4 * self.__I * self.__T,
                              self.__I**2 * self.__T * (self.__T-1)//2 + 3 * self.__I * self.__T))

        # constraint n(i,t) - y(i,t) + z(i,t) > 0
        row_ineq_idx = 0  # initial row idx for inequalities
        for t in range(self.__T):
            for i in range(self.__I):
                self.__LM[row_ineq_idx, self.N(i, t)] = self.__LM[row_ineq_idx, self.Z(i, t)] = -1
                self.__LM[row_ineq_idx, self.Y(i, t)] = 1.
                row_ineq_idx += 1

        # constraint x_i,j,t,u >= 0
        for i in range(self.__I):
            for j in range(self.__I):
                for t in range(self.__T-1):  # -1 is important here
                    for u in range(t+1, self.__T):
                        self.__LM[row_ineq_idx, self.X(i, j, t, u)] = - 1
                        row_ineq_idx += 1

        # constraints n_i,t and y_i,t and z_i,t >= 0
        for t in range(self.__T):
            for i in range(self.__I):
                self.__LM[row_ineq_idx, self.Y(i, t)] = - 1.
                row_ineq_idx += 1
                self.__LM[row_ineq_idx, self.Z(i, t)] = - 1.
                row_ineq_idx += 1
                self.__LM[row_ineq_idx, self.N(i, t)] = - 1.
                row_ineq_idx += 1

        return self.__LM

    @property
    def EMVMat(self):
        """
        Sets the equality matrix and equality vector.

        """

        if (self.__EM is not None) and (self.__EV is not None):
            return self.__EM, self.__EV

        # equality matrix
        self.__EM = np.zeros((self.__I * self.__T + self.__T - 1,
                            self.__I**2 * self.__T * (self.__T-1)//2 + 3 * self.__I * self.__T))

        # equality vector
        self.__EV = np.zeros((self.__I * self.__T + self.__T - 1, 1))

        row_eq_idx = 0  # row indx. for equalities

        # initial setting of N 
        for i in range(self.__I):
            self.__EM[row_eq_idx, self.N(i, 0)] = 1
            self.__EV[row_eq_idx] = self.N_init[i]
            row_eq_idx += 1

        # sum_i n_i,t = K 
        for t in range(1, self.__T):  # t=0 already given above
            for i in range(self.__I):
                self.__EM[row_eq_idx, self.N(i, t)] = 1
            self.__EV[row_eq_idx] = self.__K
            row_eq_idx += 1

        # constraint n_i,t = n_i,t-1 + sum + sum
        for t in range(1, self.__T):
            for i in range(self.__I):
                self.__EM[row_eq_idx, self.N(i, t)] = 1
                self.__EM[row_eq_idx, self.N(i, t-1)] = -1  # set up
                for j in range(self.__I):
                    for u in range(t):
                        self.__EM[row_eq_idx, self.X(j, i, u, t)] = -1
                    for u in range(t+1, self.__T):
                        self.__EM[row_eq_idx, self.X(i, j, t, u)] = 1
                row_eq_idx += 1

        return self.__EM, self.__EV

    @property
    def fVec(self):
        """
        Setting the optimization vector.

        """

        f_vec = np.zeros(self.__I ** 2 * self.__T * (self.__T - 1) // 2 + 3 * self.__I * self.__T)

        for t in range(self.__T):  # time period
            for i in range(self.__I):  # cities
                f_vec[self.Z(i, t)] =   self.fwd_curves[self.nbs_to_locs[i]][t]
                f_vec[self.Y(i, t)] = - self.fwd_curves[self.nbs_to_locs[i]][t]
                for j in range(self.__I):
                    for u in range(t+1, self.__T):  # CHECK IF THIS IS t+1 or NOT
                        f_vec[self.X(i, j, t, u)] = self.V[t][u][i, j]

        return f_vec

    def freightHedge(self):
        """
        Find optimum freight hedge.

        """

        EM, EV = self.EMVMat

        result =linprog( self.fVec
                       , A_ub   = self.LMMat
                       , A_eq   = EM
                       , b_ub   = np.zeros((self.__I**2 * self.__T * (self.__T-1)//2 + 4 * self.__I * self.__T, 1))
                       , b_eq   = EV
                       , bounds = list(zip(self.lowerBound, [None] * len(self.lowerBound) ) ) )

        return result.x  # actual result
