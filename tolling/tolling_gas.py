# Tolling model

import numpy as np
import multiprocessing
import ctypes
import logging

# my modules
import lattice
from cond_prob import transition_mtx_ln_blocks_fast, transition_mtx_ln_blocks_fast_internal
import sg
from pricers import cdf_vec, bvnd

# multi-threading version of tensor product
tens_fast_mt_raw = ctypes.CDLL("/home/brumen/workspace/mrds/tp.so").tensor_prod_2

logger = logging.getLogger(__name__)


def tens_fast_mt(P_m, H_m, G_m, res_m):
    tens_fast_mt_raw(ctypes.py_object(P_m),
                     ctypes.py_object(H_m),
                     ctypes.py_object(G_m),
                     ctypes.py_object(res_m))


# wrapper for the skew MRD model calibration function
def transition_wrap(arg, **kwarg):
    return tolling_model_MRD.transition_mtx_optim(*arg, **kwarg)


def step_wrap(arg, **kwarg):
    return tolling_model_MRD.step(*arg, **kwarg)


class TollingModelLatticeGas:
    """ Backward induction algorithm on the lattice
    """

    def __init__(self, params, blocks, nb_months, sg_level=15):
        """
            # params: parameters, which contain
        #   .cuda ... indicator whether CUDA is present or not
        #   .F0 ... forward price
        #   .K  ... cost of running a PP for that month
        #   .sigma_F ... forward vol
        #   .sigma_C ... cash vol
        #   .SC ... fixed startup costs
        """

        self.nb_months = nb_months
        self.cuda = params.cuda
        self.tolling_fast = params.tolling_fast # using the fast tensor routine from tolling_fast.pyx
        self.tolling_fast_mt = params.tolling_fast_mt # using raw multi-threading 
        self.fuel_ind = params.fuel_ind # whether fuel is present
        self.keep_decisions = params.keep_decisions
        self.params = params
        self.lattice_size = params.lattice_size
        self.MDT = params.MDT
        self.MUT = params.MUT
        self.maxStarts = params.maxStarts
        self.maxCap = params.maxCap
        self.minCap = params.minCap
        self.fixedStartupCost = params.fixedStartupCost
        self.fixedStartupCostCold = params.fixedStartupCostCold # NOT YET IMPLEMENTED NOT YET IMPLEMENTED
        self.fixedShutdownCost = params.fixedShutdownCost
        self.rampDnCost = params.rampDnCost
        self.rampUpCost = params.rampUpCost

        # integrating functions
        self.sg_level = sg_level
        self.sg_weights = np.array(sg.sg_w(1, self.sg_level))  # row vector
        self.sg_weights_col = self.sg_weights.reshape((self.sg_weights.size, 1))
        self.sg_xs = np.array(sg.sg_p(1, self.sg_level)).flatten()  # row vector
        self.sg_xs_col = self.sg_xs.reshape((self.sg_xs.size, 1))

        if self.fuel_ind:
            self.HR = params.HR

        # blocks
        self.blocks = blocks
        # construction of lattice blocks for each market, should eventually replace above
        self.lattice_blocks_by_name = {}

        # internal variables
        self.__marketByName = None

    @property
    def _marketByName(self):
        """

        """

        if self.__marketByName:
            return self.__marketByName

        for dp in self.blocks.market.power:
            for m in dp:
                market_name = m["name"]
                if market_name not in self.lattice_blocks_by_name:
                    self.lattice_blocks_by_name[market_name] = lattice.lattice_ln(m["fwd"],
                                                                                  m["sigma_F"],
                                                                                  m["sigma_C"],
                                                                                  self.lattice_size).lattice
                if market_name not in self.__marketByName:
                    self.__marketByName[market_name] = m

    @property
    def _latticeBlocks(self):
        # lattice per blocks; done so that it uses pointers
        return [[self.lattice_blocks_by_name[m["name"]]
                 for m in dp]
                 for dp in self.blocks.market.power]

        if self.fuel_ind:  # gas lattice is only used if fuel is present
            self.lattice_gas = lattice.lattice_ln(self.blocks.market.gas["fwd"],
                                                  self.blocks.market.gas["sigma_F"],
                                                  self.blocks.market.gas["sigma_C"],
                                                  self.lattice_size).lattice

        blocks_seqs = self.construct_hours()
        self.hours_seq = blocks_seqs["hours_seq"]
        self.t_diff_seq = blocks_seqs["blocks_tdiff"]
        self.market_seq = blocks_seqs["market_seq"]  # sequence of market moves
        self.lattice_seq = blocks_seqs["lattice_seq"]
        self.P_m_seq = {}  # sequence of transition matrices, currently empty, filled in as the program progresses
        self.zero_matrix = self._zeroPP()  # profit matrix/vector

        def list_zero_pp(nb):
            return [self._zeroPP()] * nb

        if self.cuda:
            self.zero_matrix_d = gpa.to_gpu(self._zeroPP()).astype(np.float32)

        # tmp. storage 
        if not self.cuda:
            self.work_curr_tmp = {"max": self._zeroPP(), "min": self._zeroPP()} # work between max and min capacity
            self.idle_curr_tmp = self._zeroPP()
        else:
            self.work_curr_tmp = gpa.to_gpu(self._zeroPP()).astype(np.float32)
            self.idle_curr_tmp = gpa.to_gpu(self._zeroPP()).astype(np.float32)

        # the next power plant (power_prices) working condition
        self.work_pp_next = {"max": list_zero_pp(self.MUT+1),
                             "min": list_zero_pp(self.MUT+1)}
        self.work_pp_curr = {"max": list_zero_pp(self.MUT+1),
                             "min": list_zero_pp(self.MUT+1)}
        self.idle_pp_next = list_zero_pp(self.MDT+1)
        self.idle_pp_curr = list_zero_pp(self.MDT+1)

        # even if we dont keep decisions, we just dont use these
        self.work_pp_curr_ind = [{"max": [], "min": []}] * 60
        self.idle_pp_curr_ind = [[]] * 60  # IMPROVE ON 60

        if self.keep_decisions:
            for block in range(60):
                self.work_pp_curr_ind[block]['max'] = [self._zeroPP() for n in range(self.MUT + 1)]
                self.work_pp_curr_ind[block]["min"] = [self._zeroPP() for n in range(self.MUT + 1)]
                self.idle_pp_curr_ind[block] = [self._zeroPP() for n in range(self.MDT + 1)]
        else:
            for block in range(60):
                self.work_pp_curr_ind[block]['max'] = [0 for n in range(self.MUT + 1)]
                self.work_pp_curr_ind[block]["min"] = [0 for n in range(self.MUT + 1)]
                self.idle_pp_curr_ind[block] = [0 for n in range(self.MDT + 1)]

        for Ut in range(self.MUT + 1):
            if not self.cuda:
                self.work_pp_next["max"].append(self._zeroPP())
                self.work_pp_curr["max"].append(self._zeroPP())
                self.work_pp_next["min"].append(self._zeroPP())
                self.work_pp_curr["min"].append(self._zeroPP())
            else:
                self.work_pp_next["max"].append(gpa.to_gpu(self._zeroPP()).astype(np.float32))
                self.work_pp_curr["max"].append(gpa.to_gpu(self._zeroPP()).astype(np.float32))
                self.work_pp_next["min"].append(gpa.to_gpu(self._zeroPP()).astype(np.float32))
                self.work_pp_curr["min"].append(gpa.to_gpu(self._zeroPP()).astype(np.float32))

        for Dt in range(self.MDT + 1):
            if not self.cuda:
                self.idle_pp_next.append(self._zeroPP())
                self.idle_pp_curr.append(self._zeroPP())
            else:
                self.idle_pp_next.append(gpa.to_gpu(self._zeroPP()).astype(np.float32))
                self.idle_pp_curr.append(gpa.to_gpu(self._zeroPP()).astype(np.float32))

    def _zeroPP(self):
        """
        Construction of the zero matrix

        """

        if self.fuel_ind:
            return np.zeros((self.lattice_size, self.lattice_size))  # matrix

        return np.zeros(self.lattice_size) # vector

    def p_t( self
           , v
           , p_dash
           , g_dash
           , op
           , g
           , F_123
           , sigma_123
           , rho_123
           , t
           , delta_t ):
        """
        transition prob of (p_dash, g_dash | op, g)
        v replaces network_struct via equation (that can be given
        """

        F_1, F_2, F_3 = F_123
        rho_12, rho_13, rho_23 = rho_123
        sigma_P, sigma_OP, sigma_G = sigma_123
        sigma_1, sigma_2, sigma_3 = sigma_123

        sigma_cond = 1. - (rho_12**2 - 2 * rho_12 * rho_13 * rho_23 + rho_13**2) \
            / (1. - rho_23**2)

        # d1 = (np.log(network_struct/F_1) + 0.5 * sigma_1**2 * t) / (sigma_1 * np.sqrt(t))
        d2 = (np.log(op/F_2) + 0.5 * sigma_2**2 * t) / (sigma_2 * np.sqrt(t))
        d3 = (np.log(g/F_3) + 0.5 * sigma_3**2 * t) / (sigma_3 * np.sqrt(t))

        mu_over = (rho_12 * d2 - rho_12 * rho_23 * d3 - rho_13 * rho_23 * d2 + rho_13 * d3) \
            / (1. - rho_23**2)

        d4 = (np.log(p_dash/F_1) + 0.5 * sigma_P**2 * (t + delta_t)
            - np.sqrt(sigma_cond) * v * sigma_P * np.sqrt(t) +
            mu_over * sigma_P * np.sqrt(t)) / (sigma_P * np.sqrt(delta_t))
        d5 = (np.log(g_dash/g) + 0.5 * sigma_G**2 * delta_t) / (sigma_G * np.sqrt(delta_t))

        return bvnd(d4, d5, rho_13)  # rho between Peak & gas (peak is mnemonic for next one)

    def p_t_integ( self
                 , p_dash
                 , g_dash
                 , op
                 , g
                 , F_123
                 , sigma_123
                 , rho_123
                 , t
                 , delta_t
                 , sg_level=15 ):

        weights = np.array(sg.sg_w(1, sg_level))  # row vector
        xs      = np.array(sg.sg_p(1, sg_level)).flatten()  # row vector

        g_v = self.p_t( xs * np.sqrt(2.)
                      , p_dash
                      , g_dash
                      , op
                      , g
                      , F_123
                      , sigma_123
                      , rho_123
                      , t
                      , delta_t ) / np.sqrt(np.pi)  # g = lambda x: f(x * sqrt2) / (sqrt_pi)**D

        return np.sum(weights * g_v)

    def transition_mtx_ln_blocks_all(self, step_nb):
        """
        constructs a transition matrix P_m for step step_nb

        """

        if step_nb == len(self.hours_seq) - 1:  # last step, no transition
            return np.zeros((self.lattice_size, self.lattice_size,
                             self.lattice_size, self.lattice_size))

        else:
            P_next_v = self.lattice_seq[step_nb]  # next power lattice
            G_next_v = self.lattice_gas  # next gas lattice, col. vec.
            P_curr_v = self.lattice_seq[step_nb+1]  # curr. power lattice
            G_curr_v = self.lattice_gas  # curr. gas lattice
            P_m = np.zeros ((self.lattice_size, self.lattice_size,
                             self.lattice_size, self.lattice_size))

            # IMPROVE IMPROVE IMPROVE
            P_m_tmp = np.zeros(self.lattice_size + 1)  # tmp. mtx for taking differences

            # v, p_dash, g_dash, op, g, F_123, sigma_123, rho_123, t, delta_t = 1

            # for reference
            # def p_t_integ(p_dash, g_dash, op, g, F_123, sigma_123, rho_123, t, delta_t,
            #          sg_level = 15):
            P_m_tmp = np.zeros((self.lattice_size, self.lattice_size,
                                self.lattice_size, self.lattice_size))

            fwd_vals = [self.market_seq[step_nb+1]["fwd"],
                        self.market_seq[step_nb]  ["fwd"],
                        self.blocks.market.gas    ["fwd"] ]
            sigma_vals = [self.market_seq[step_nb+1]["sigma_C"],
                          self.market_seq[step_nb]  ["sigma_C"],
                          self.blocks.market.gas    ["sigma_C"] ]

            corr_vals = [0.8, 0.9, 0.95]  # correlations WRONG WRONG WRONG

            for (P_curr_idx, P_curr) in enumerate(P_curr_v):
                for (G_curr_idx, G_curr) in enumerate(G_curr_v):
                    for (P_next_idx, P_next) in enumerate(P_next_v):
                        for (G_next_idx, G_next) in enumerate(G_next_v):
                            P_m[P_curr_idx, G_curr_idx, P_next_idx, G_next_idx] = \
                            self.p_t_integ(P_next, G_next, P_curr, G_curr,
                                      fwd_vals, sigma_vals, corr_vals,
                                      np.sum(self.t_diff_seq[:step_nb+2]),
                                      self.t_diff_seq[step_nb+2])  # WRONG WRONG WRONG

            # now that the distribution function is constructed, density is needed.
            for (P_curr_idx, P_curr) in enumerate(P_curr_v):
                for (G_curr_idx, G_curr) in enumerate(G_curr_v):
                    P_m_tmp[P_curr_idx, G_curr_idx, 1:, :] = P_m[P_curr_idx, G_curr_idx, 1:, :] - \
                        P_m[P_curr_idx, G_curr_idx, :-1, :]
                    P_m_tmp[P_curr_idx, G_curr_idx, :, 1:] = P_m_tmp[P_curr_idx, G_curr_idx, :, 1:] - \
                        P_m[P_curr_idx, G_curr_idx, :, :-1]
                    P_m_tmp[P_curr_idx, G_curr_idx, 1:, 1:] = P_m_tmp[P_curr_idx, G_curr_idx, 1:, 1:] + \
                        P_m[P_curr_idx, G_curr_idx, :-1, :-1]

            # if np.abs(P_m[F_curr_idx,-1])> 1e-2:
            #    P_m_tmp[1:] = P_m[F_curr_idx,:] / P_m[F_curr_idx,-1] # normalization - NOT SURE IF RIGHT??????????
            # P_m[F_curr_idx,:] = np.diff(P_m_tmp)

            return P_m_tmp

    def transit_val(self, P_m, H_m, curr_profit_m):
        """ Transition value of tolling:
        :param P_m: transition matrix (or tensor)
        :param H_m: next value of tolling given the lattice
        :param curr_profit_m: running profit
        """

        res_m = self._zeroPP()

        if self.fuel_ind:
            power_lattice_size, gas_lattice_size = P_m.shape[0:2]
            for P_curr_idx in range(power_lattice_size):
                for G_curr_idx in range(gas_lattice_size):
                    if isinstance(curr_profit_m, np.ndarray):
                        profit_curr = curr_profit_m[P_curr_idx, G_curr_idx]
                    else:  # const. profit (such as shutdown costs)
                        profit_curr = curr_profit_m
                    res_m[P_curr_idx, G_curr_idx] = np.sum(P_m[P_curr_idx, G_curr_idx, :, :] * H_m) + profit_curr
        else:
            res_m = np.dot(P_m, H_m) + curr_profit_m

        return res_m

    def construct_hours(self):
        """ Construct the hours from the blocks structure
        """

        days = np.array([0.])
        hours_seq = []
        market_seq = []
        lattice_seq = []
        for day in range(30):
            day_week = np.mod(day, 7)
            hours_market_for_day_week = [(hp, mp, lb) for (hp, dp, mp, lb) in
                                         zip(self.blocks.hours_partition, self.blocks.days_partition,
                                             self.blocks.market.power, self._latticeBlocks) if day_week in dp][0]
            hours_for_day_week, market_for_day_week, lattice_for_day_week = hours_market_for_day_week
            days = np.append(days, days[-1] + np.cumsum(hours_for_day_week) / (24. * 365.))
            hours_seq.extend(hours_for_day_week)
            market_seq.extend(market_for_day_week)
            lattice_seq.extend(lattice_for_day_week)

        self.nb_steps = len(hours_seq)  # number of steps for the lattice

        days_diff = np.zeros(len(days))  # CHANGE HERE CAHNGE CHANGE
        days_diff[1:] = np.diff(days)

        return { 'blocks_tdiff': days_diff
               , 'hours_seq'   : hours_seq
               , 'market_seq'  : market_seq
               , 'lattice_seq' : lattice_seq }

    def running_profit(self, block_nb):
        """
        computes the running profit in block block_nb
        """

        power = self.lattice_seq[block_nb]
        power_l = len(power)
        power_mtx = np.kron(power.reshape((power_l, 1)), np.ones(len(self.lattice_gas)))
        gas_mtx = np.kron(np.ones((power_l, 1)), self.lattice_gas)
        hr_profit = power_mtx - self.HR * gas_mtx
        profit_cap_indep = hr_profit * self.hours_seq[block_nb]

        # profit_max_cap, profit min capacity
        return profit_cap_indep * self.maxCap, profit_cap_indep * self.minCap

    def step(self, Ut_curr, Dt_curr, block_nb, start_nb=10):
        """
        :param Ut_curr: current uptime of the PP
        :param Dt_curr: current downtime of the PP
        :param block_nb: which block nb_sims in the month are we currently at
        :param start_nb: number of startups of the PP in that month NOT IMPLEMENTED YET
        """

        profit_max, profit_min = self.running_profit(block_nb)

        if block_nb not in self.P_m_seq:
            self.P_m_seq[block_nb] = self.transition_mtx_ln_blocks_all(block_nb)

        Pm = self.P_m_seq[block_nb]  # current transition matrix

        if Ut_curr == self.MUT:  # minimum uptime reached; options: ramp up/down, switch off
            self.work_pp_curr["max"][Ut_curr] = np.maximum(
                    np.maximum(
                        self.transit_val(Pm, self.work_pp_next["max"][self.MUT], profit_max),
                        self.transit_val(Pm, self.work_pp_next["min"][self.MUT], profit_max) - self.rampDnCost),
                    self.transit_val(Pm, self.idle_pp_next[0], profit_max - self.fixedShutdownCost)
            )
            self.work_pp_curr["min"][Ut_curr] = np.maximum(
                    np.maximum(
                        self.transit_val(Pm, self.work_pp_next["min"][self.MUT], profit_min),
                        self.transit_val(Pm, self.work_pp_next["max"][self.MUT], profit_min) - self.rampUpCost),
                    self.transit_val(Pm, self.idle_pp_next[0], profit_min - self.fixedShutdownCost))
        else:  # MUT not reached; we can not switch off, just ramp up/down
            self.work_pp_curr["max"][Ut_curr] = np.maximum(
                    self.transit_val(Pm, self.work_pp_next["max"][Ut_curr + 1], profit_max),
                    self.transit_val(Pm, self.work_pp_next["min"][Ut_curr + 1], profit_max) - self.rampDnCost)
            self.work_pp_curr["min"][Ut_curr] = np.maximum(
                    self.transit_val(Pm, self.work_pp_next["min"][Ut_curr + 1], profit_min),
                    self.transit_val(Pm, self.work_pp_next["max"][Ut_curr + 1], profit_min) - self.rampUpCost)

        if Dt_curr == self.MDT:  # min. downtime reached; can be restarted
            self.idle_pp_curr[Dt_curr] = np.maximum(
                    np.maximum(
                        self.transit_val(Pm, self.idle_pp_next[self.MDT], self.zero_matrix),  # cont. idle
                        self.transit_val(Pm, self.work_pp_next["max"][0], - self.fixedStartupCost)),  # restart to max
                self.transit_val(Pm, self.work_pp_next["min"][0], - self.fixedStartupCost))
        else:  # can not be restarted, just run idle
            self.idle_pp_curr[Dt_curr] = self.transit_val(Pm, self.idle_pp_next[Dt_curr + 1], self.zero_matrix)

    # all steps for 1 time step 
    def all_one_steps(self, block_nb):
        if not self.cuda:
            if self.tolling_fast:  # fast version of tensor product
                self.step_fast(0, 0)
            elif self.tolling_fast_mt:
                self.step_fast_mt(0, 0)
            else:
                self.step(0, 0, block_nb)
        else:
            self.step_cuda(0, 0)

        for Ut in range(1, self.MUT + 1):
            if not self.cuda:
                if self.tolling_fast:
                    self.step_fast(Ut, 0)
                elif self.tolling_fast_mt:
                    self.step_fast_mt(Ut, 0)
                else:
                    self.step(Ut, 0, block_nb)
            else:
                self.step_cuda(Ut, 0)
        for Dt in range(1, self.MDT + 1):
            if not self.cuda:
                if self.tolling_fast:
                    self.step_fast(0, Dt)
                elif self.tolling_fast_mt:
                    self.step_fast_mt(0, Dt)
                else:
                    self.step(0, Dt, block_nb)
            else:
                self.step_cuda(0, Dt)

    def all_one_steps_mt(self):
        """ mult-ithreading version of the step function
        """

        nb_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=nb_cores)
        pool.map(step_wrap,
                 zip([self] * (self.MUT + 1), range(self.MUT + 1),
                     [0] * (self.MUT + 1)))
        pool.map(step_wrap,
                 zip([self] * (self.MDT + 1),
                     [0] * (self.MDT + 1), range(self.MDT + 1)))

    def overwrite_next_w_curr(self):
        """
        generates work_pp_next, idle_pp_next from the current
        """
        for work_pp_ind in range(self.MUT+1):
            self.work_pp_next["max"][work_pp_ind] = self.work_pp_curr["max"][work_pp_ind]
            self.work_pp_next["min"][work_pp_ind] = self.work_pp_curr["min"][work_pp_ind]
        for idle_pp_ind in range(self.MDT+1):
            self.idle_pp_next[idle_pp_ind] = self.idle_pp_curr[idle_pp_ind]

    def multiple_steps(self, nb_blocks):
        for block_nb in range(nb_blocks - 1, -1, -1):  # walking over blocks
            logger.info("Block {0} of {1}".format(block_nb, nb_blocks))
            self.all_one_steps(block_nb)
            self.overwrite_next_w_curr()

    def tolling_value(self, month):
        """ compute the tolling value from partial tolls for the given month

        :param month: month for which the tolling is to be computed.
        """

        self.multiple_steps(self.nb_steps)  # do the steps within the month

        if not self.cuda:
            # compute average over the values here self.work_pp_curr[0]
            F_init = self.lattice_seq[0]
            F_0_init = self.market_seq[0]["fwd"]
            sigma_F_init = self.market_seq[0]["sigma_F"]
            T_m_init = self.blocks[month].Tm

            cdf_Tm = cdf_vec((np.log(F_init / F_0_init ) + 0.5 * sigma_F_init ** 2 * T_m_init ) /
                             (sigma_F_init * np.sqrt(T_m_init)))
            cdf_Tm_a = np.zeros(self.lattice_size + 1)
            cdf_Tm_a[1:] = cdf_Tm
            pdf_Tm = np.diff(cdf_Tm_a)
            best_strategy = np.maximum (np.maximum(self.work_pp_curr["min"][0] - self.fixedStartupCost,
                                                   self.work_pp_curr["max"][0] - self.fixedStartupCost),
                                        self.idle_pp_curr[0])
            return np.sum(pdf_Tm * best_strategy)

        # cuda version.
        return self.work_pp_curr[0].get()[self.lattice_size / 2, self.lattice_size / 2]

    def compute_val(self):
        """ Compute the total tolling value using the lattice model.
        """

        return sum([self.tolling_value(month) for month in range(self.nb_months)])
