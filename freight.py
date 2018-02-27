# Freight model implementation 
#
import config
import numpy as np
import openopt
import pricers  # for bivariate options


class Freight:
    """
    Freight class:
      N ... initial distribution of the fleet ... IN WHICH FORMAT SHOULD THIS BE?
      K ... number of tankers ... CHECK CHECK CHECK
      I ... number of cities
      T ... number of time periods
    """
    def __init__(self, fwd_date, locs, fwd_curves, vol_curves, corr_mtx, travel_mtx,
                 N_init, T):
        """
        initial variables are like in test_freights
        """
        self.fwd_date = fwd_date
        self.locs = locs
        self.nb_locs = len(locs)  # number of locations
        self.nbs_to_locs = {idx: loc for (idx, loc) in zip(range(len(locs)), locs)}
        self.locs_to_nbs = {loc: idx for (idx, loc) in zip(range(len(locs)), locs)}
        self.fwd_curves = fwd_curves
        self.vol_curves = vol_curves
        self.corr_mtx = corr_mtx
        self.travel_mtx = travel_mtx  # number of periods between different locations
        self.K = np.sum(N_init)  # number of tankers
        self.T = T  # number of time periods
        self.N_init = N_init  # init. distrib. of fleet
        self.I = len(locs)  # number of cities
        I = len(N_init)  # local variable

        # initialization of variables (THIS CAN BE DONE IN SPARSE MATRICES)
        # >= 0 matrix and vector
        self.LM = np.zeros((I**2 * T * (T-1)/2 + 4 * I * T,
                            I**2 * T * (T-1)/2 + 3 * I * T))
        self.LV = np.zeros((I**2 * T * (T-1)/2 + 4 * I * T, 1))
        # = matrix and vector 
        self.EM = np.zeros((I * T + T - 1,
                            I**2 * T * (T-1)/2 + 3 * I * T))
        self.EV = np.zeros((I * T + T - 1, 1))

        self.f_vec = np.zeros(I**2 * T * (T-1)/2 + 3 * I * T)  # optimizing vec
        self.lb = np.zeros(I**2 * T * (T-1)/2 + 3*I*T)  # lower boundary

        # empty list of V (=value matrices)
        self.V = [[0] * self.T] * self.T
        self.result = None

    def construct_matrices(self):
        """
        construction of spread option matrices
        V[t_1][t_2] [c_1, c_2], t_1 < t_2
        """
        # empty list of V
        self.V = [[0] * self.T] * self.T  # by time periods
        fcl = self.fwd_curves  # just abbreviations
        avl = self.vol_curves
        vf = pricers.spread_option_kirk  # valuation function
        # V construction
        for t_1 in range(self.T):
            for t_2 in range((t_1+1), self.T):
                self.V[t_1][t_2] = np.zeros((self.nb_locs, self.nb_locs))
                for c_1_nb, c_1 in enumerate(self.locs):  # city 1
                    for c_2_nb, c_2 in enumerate(self.locs):  # city 2
                        if self.corr_mtx.has_key((c_1, c_2)):
                            c1_used, c2_used = c_1, c_2
                        else:
                            c1_used, c2_used = c_2, c_1
                        self.V[t_1][t_2][c_1_nb, c_2_nb] = vf(fcl[c_1][t_1], fcl[c_2][t_2], 0.,
                                                              avl[c_1][t_1], avl[c_2][t_2],
                                                              self.corr_mtx[(c1_used, c2_used)],
                                                              (t_2 - t_1), 0.99)  # DISCOUNT FACTOR IS WRONG
                        # spread_option_kirk (F_1, F_2, K, sigma_1, sigma_2, rho, T, DF)

    def X(self, i, j, t, u):
        """
        # index in a matrix of a x, y, z, n (see document)
        """
        return i + j * self.I + self.I**2 * \
            (self.T - 1 - u + (self.T * t - t * (t+1)/2))

    def X_inv(self, n):
        """
        extracts the i, j, t, u from n
        """
        i = n % self.I
        j = (n % self.I**2 - i)/self.I
        u_v = n / self.I**2
        u, v = 0, 0   # THIS IS WRONG WRONG WRONG
        return i, j, u, v

    def Y(self, i, t):
        return self.I**2 * self.T * (self.T - 1)/2 + i + t * self.I

    def Z(self, i, t):
        return self.I**2 * self.T * (self.T - 1)/2 + self.I * self.T + \
            i + t * self.I

    def N(self, i, t):
        return self.I**2 * self.T * (self.T - 1)/2 + 2*self.I*self.T + \
            i + t * self.I

    def ineq(self):
        """
        sets the inequality matrix LM
        """
        # constraint n(i,t) - y(i,t) + z(i,t) > 0
        row_ineq_idx = 0  # initial row idx for inequalities
        for t in range(self.T):
            for i in range(self.I):
                self.LM[row_ineq_idx, self.N(i, t)] = self.LM[row_ineq_idx, self.Z(i, t)] = -1
                self.LM[row_ineq_idx, self.Y(i, t)] = 1
                row_ineq_idx += 1

        # constraint x_i,j,t,u >= 0
        for i in range(self.I):
            for j in range(self.I):
                for t in range(self.T-1):  # -1 is important here
                    for u in range(t+1, self.T):
                        self.LM[row_ineq_idx, self.X(i, j, t, u)] = - 1
                        row_ineq_idx += 1

        # constraints n_i,t and y_i,t and z_i,t >= 0
        for t in range(self.T):
            for i in range(self.I):
                self.LM[row_ineq_idx, self.Y(i, t)] = - 1
                row_ineq_idx += 1
                self.LM[row_ineq_idx, self.Z(i, t)] = - 1
                row_ineq_idx += 1
                self.LM[row_ineq_idx, self.N(i, t)] = - 1
                row_ineq_idx += 1

    def eq(self):
        """
        # sets the equality matrix
        """
        row_eq_idx = 0  # row indx. for equalities

        # initial setting of N 
        for i in range(self.I):
            self.EM[row_eq_idx, self.N(i, 0)] = 1
            self.EV[row_eq_idx] = self.N_init[i]
            row_eq_idx += 1

        # sum_i n_i,t = K 
        for t in range(1, self.T):  # t=0 already given above
            for i in range(self.I):
                self.EM[row_eq_idx, self.N(i, t)] = 1
            self.EV[row_eq_idx] = self.K 
            row_eq_idx += 1

        # constraint n_i,t = n_i,t-1 + sum + sum
        for t in range(1, self.T):
            for i in range(self.I):
                self.EM[row_eq_idx, self.N(i, t)] = 1
                self.EM[row_eq_idx, self.N(i, t-1)] = -1  # set up
                for j in range(self.I):
                    for u in range(t):
                        self.EM[row_eq_idx, self.X(j, i, u, t)] = -1
                    for u in range(t+1, self.T):
                        self.EM[row_eq_idx, self.X(i, j, t, u)] = 1
                row_eq_idx += 1

    def construct_f_vec(self):
        """
        setting the optimization vector
        """
        for t in range(self.T):  # time period
            for i in range(self.I):  # cities
                self.f_vec[self.Z(i, t)] = self.fwd_curves[self.nbs_to_locs[i]][t]
                self.f_vec[self.Y(i, t)] = - self.fwd_curves[self.nbs_to_locs[i]][t]
                for j in range(self.I):
                    for u in range(t+1, self.T):  # CHECK IF THIS IS t+1 or NOT
                        self.f_vec[self.X(i, j, t, u)] = self.V[t][u][i, j]

    def solve(self):
        # USE LB AND UB, WILL MAKE IT FASTER
        # SHOULD BE - IN FRONT OF F_VEC
        p = openopt.LP(self.f_vec, A=self.LM, Aeq=self.EM, b=self.LV, beq=self.EV, lb=self.lb)
        # p = openopt.MILP(self.f_vec, A=self.LM, intVars=range(len(self.f_vec)),
        #                 Aeq=self.EM, b=self.LV, beq=self.EV, lb=self.lb)
        r = p.solve('cvxopt_lp')  # MAXIMIZE OR MINIMIZE???
        # r = p.solve('glpk')  # MAXIMIZE OR MINIMIZE???
        p.debug = 1
        return r.xf

    def set_n_solve(self):
        self.construct_matrices()
        self.ineq()
        self.eq()
        self.construct_f_vec()
        self.result = self.solve()
        return self.result
