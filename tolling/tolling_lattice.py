# independent tolling model
from config import CUDA_PRESENT
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
if CUDA_PRESENT:
    import pycuda.autoinit
    import pycuda.gpuarray as gpa
    import cublas
    import pycuda.cumath


# my modules
import lattice
from pricers.pricers import cdf_vec
from cond_prob import transition_mtx_ln_blocks_fast, transition_mtx_ln_blocks_fast_internal
import sg  # sparse grids for fast integration

if CUDA_PRESENT:
    import cuda_ops as co

import logging

logger = logging.Logger(__name__)


def compute_val_one_month_wrap(arg, **kwarg):
    """
    helper function for multiprocessing (compute_val_one_month)
    """

    o, m = arg
    return o.compute_val_one_month(m)


class TollingModelLattice(object):
    """
    Backward induction tolling algorithm, based on the lattice.

    """

    def __init__( self
                , params
                , blocks
                , debug_ind = False
                , keep_dec  = False
                , cuda_ind  = False
                , sg_level  = 15
                , dtype_used= np.float ):
        """
        backward induction algorithm on the lattice

        params: parameters, which contain
          .cuda ... indicator whether CUDA is present or not
          .F0 ... forward price
          .K  ... cost of running a PP for that month
          .sigma_F ... forward vol
          .sigma_C ... cash vol
          .SC ... fixed startup costs
        debug_ind ... indicator whether debug is desired
        """

        self.tolling_fast = params.tolling_fast  # using the fast tensor routine from tolling_fast.pyx
        self.tolling_fast_mt = params.tolling_fast_mt  # using raw multi-threading
        self.params = params  # THIS CAN PERHAPS BE REMOVED LATER
        self.lattice_size = params.lattice_size
        self.MDT = params.MDT  # integer, number of blocks of minimum downtime
        self.MUT = params.MUT  # integer, minimum uptime
        self.maxStarts = params.maxStarts  # integer
        self.max_cap = params.maxCap  # maximum capacity
        self.min_cap = params.minCap
        self.fixed_startup_cost = params.fixedStartupCost
        self.fixed_startup_cost_cold = params.fixedStartupCostCold  # not yet implemented
        self.fixed_sd_cost = params.fixedShutdownCost
        self.ramp_dn_cost = params.rampDnCost
        self.ramp_up_cost = params.rampUpCost
        self.keep_dec = keep_dec  # inidicator whether to keep decisions
        self.cuda = cuda_ind  # cuda indicator
        self.dtype_used = dtype_used

        # blocks
        self.blocks = blocks
        # construction of lattice blocks for each market, should eventually replace above
        self.lattice_blocks_by_name = {}
        self.market_by_name = {}
        for dp in self.blocks.market:
            for m in dp:
                market_name = m["name"]
                if market_name not in self.lattice_blocks_by_name:
                    self.lattice_blocks_by_name[market_name] = lattice.lattice_ln(m["fwd"],
                                                                                  m["sigma_F"],
                                                                                  m["sigma_C"],
                                                                                  self.lattice_size,
                                                                                  ci_ind=self.cuda).lattice
                if not self.market_by_name.has_key(market_name):
                    self.market_by_name[market_name] = m

        # lattice per blocks; done so that it uses pointers
        self.lattice_blocks = [[self.lattice_blocks_by_name[m["name"]] for m in dp]
                               for dp in self.blocks.market]

        hbml = self.construct_hours()  # hours-blocks-market-lattice
        self.hours_seq = hbml["hours_seq"]
        self.t_diff_seq = hbml["blocks_tdiff"]
        self.market_seq = hbml["market_seq"]  # sequence of market moves
        self.lattice_seq = hbml["lattice_seq"]

        self.P_m_seq = {}  # sequence of transition matrices, currently empty, filled in as the program progresses
        self.fuel_ind = False  # whether fuel is present
        self.debug_ind = debug_ind
        # profit matrix/vector
        self.zero_matrix = self.zero_pp()

        # the next power plant (power_prices) working condition
        self.work_pp_next = {"max": [], "min": []}  # max for max capacity, min for min cap.; only 2 states for now
        self.work_pp_curr = {"max": [], "min": []}
        self.work_pp_curr_ind = [{"max": [], "min": []}] * 60  # indicator for the max, min
        self.idle_pp_next = []
        self.idle_pp_curr = []
        self.idle_pp_curr_ind = [[]] * 60
        for block in range(60):
            self.work_pp_curr_ind[block]["max"] = [self.zero_pp()] * (self.MUT + 1)
            self.work_pp_curr_ind[block]["min"] = [self.zero_pp()] * (self.MUT + 1)

        self.work_pp_next["max"] = [self.zero_pp()] * (self.MUT + 1)
        self.work_pp_curr["max"] = [self.zero_pp()] * (self.MUT + 1)
        self.work_pp_next["min"] = [self.zero_pp()] * (self.MUT + 1)
        self.work_pp_curr["min"] = [self.zero_pp()] * (self.MUT + 1)

        self.idle_pp_next = [self.zero_pp()] * (self.MDT + 1)
        self.idle_pp_curr = [self.zero_pp()] * (self.MDT + 1)

        for block in range(60):
            self.idle_pp_curr_ind[block] = [self.zero_pp()] * (self.MDT + 1)

        # sg_grid matrix
        self.sg_level = sg_level
        if not self.cuda:
            self.sg_weights = np.array(sg.sg_w(1, sg_level))
            self.sg_xs = np.array(sg.sg_p(1, sg_level)).flatten()
        else:
            self.sg_weights = gpa.to_gpu(np.array(sg.sg_w(1, sg_level)))
            self.sg_xs = gpa.to_gpu(np.array(sg.sg_p(1, sg_level)).flatten())

    def zero_pp(self):
        """ Construction of the zero matrix

        """

        if not self.cuda:
            if self.fuel_ind:
                return np.zeros((self.lattice_size, self.lattice_size))  # matrix

            return np.zeros(self.lattice_size)  # vector

        if self.fuel_ind:
            return gpa.zeros((self.lattice_size, self.lattice_size), dtype=self.dtype_used)

        return gpa.zeros(self.lattice_size, dtype=self.dtype_used)  # vector

    def maximum_2(self, x, y, keep_dec=False):
        """ computes the maximum and the indices:
          0 for 1st, 1 for 2nd)

        :param x: maximum over these and y is taken
        :type x: gpa.GPUArray or np.array
        :param y: the second vector over which max is taken
        :type y: gpa.GPUArray or np.array
        :param ci: cuda indicator
        :type ci: bool
        :param keep_dec: keep the decision variables
        :type keep_dec: bool
        """
        xy_max = gpa.maximum(x, y) if self.cuda else np.maximum(x, y)

        if keep_dec:
            xy_ind = (xy_max == y)
            return xy_max, xy_ind

        return xy_max, 0

    def maximum_3(self, x, y, z, keep_dec=False):
        """
        maximum in 3 elts, for:

          0 - for maximum in x
          1 - maximum in y
          2 - maximum in z
        :param keep_dec: keep the decision which is the maximum (useful for switch/shutdown decisions).
        """

        xy_max, xy_ind = self.maximum_2(x, y, keep_dec=keep_dec)
        xyz_max = gpa.maximum(xy_max, z) if self.cuda else xyz_max = np.maximum(xy_max, z)
        xyz_ind = (xyz_max == xy_max) * xy_ind + (xyz_max == z) * 2 if keep_dec else 0

        return xyz_max, xyz_ind

    def tm_ln_blocks_sg_fast(self, p_dash, op, F_P, F_OP,
                             sigma_P, sigma_OP, rho,
                             t, delta_t=1./365.,
                             sg_level=15):
        """
        sparse grid integration without functions
          level ... sg level, how fine the sparse grid gets
          xs ... abscisses over which the integration is done
        """
        def tm_ln_blocks_cpu(v, p_dash, op,
                             F_P, F_OP,
                             sigma_P, sigma_OP,
                             rho, t, delta_t=1./365.):
            """
            :param op: number
            :param v: row vector
            :param p_dash: column vector
            d1 is a matrix, on which d1 is computed
            """
            z_op = (np.sqrt(op / F_OP) + 0.5 * sigma_OP ** 2 * t) / (sigma_OP * np.sqrt(t))  # number
            # z_p = (np.log(network_struct/F_P) + 0.5 * sigma_P**2 * t) / (sigma_P * np.sqrt(t) )
            # v = (z_p - rho * z_op) / sqrt(1. - rho**2)
            d1 = (np.log(p_dash / F_P) + 0.5 * sigma_P ** 2 * (t + delta_t) -
                  np.sqrt(1. - rho ** 2) * sigma_P * np.sqrt(t) * v -
                  rho * z_op * sigma_P * np.sqrt(t)) / (sigma_P * np.sqrt(delta_t))

            return cdf_vec(d1, ci=False)

        def tm_ln_blocks_gpu( v
                            , p_dash
                            , op
                            , F_P
                            , F_OP
                            , sigma_P
                            , sigma_OP
                            , rho
                            , t
                            , delta_t=1./365. ):
            """
            GPU version of the function above

            :param op: number
            :param v: row vector
            :param p_dash: column vector
            d1 is a matrix, on which d1 is computed
            """

            z_op = (pycuda.cumath.sqrt(op / F_OP) + 0.5 * sigma_OP ** 2 * t) / (sigma_OP * np.sqrt(t))  # number
            col_vec = pycuda.cumath.log(p_dash / F_P) + 0.5 * sigma_P ** 2 * (t + delta_t)
            row_vec = - np.sqrt(1. - rho ** 2) * sigma_P * np.sqrt(t) * v - \
                z_op.get() * rho * sigma_P * np.sqrt(t)
            d1 = co.vtpv(col_vec, row_vec) / (sigma_P * np.sqrt(delta_t))

            return cdf_vec(d1, ci=True)

        sqrt2 = np.sqrt(2.)
        sqrt_pi = np.sqrt(np.pi)

        # p_dash has to be column vector
        # g = lambda x: f(x * sqrt2) / (sqrt_pi)**D
        g_v_fct = tm_ln_blocks_cpu if not self.cuda else tm_ln_blocks_gpu

        # g_v is a matrix
        g_v = g_v_fct((self.sg_xs * sqrt2),
                      p_dash, op, F_P, F_OP,
                      sigma_P, sigma_OP, rho, t, delta_t) / sqrt_pi
        if not self.cuda:
            return np.sum(self.sg_weights * g_v, axis=1)
        else:
            # implement self.weights * g_v (weights: row vec, g_v: mtx row x col)
            co.vtpm(self.sg_weights, g_v, tm_ind='t', new_mtx_gen=False)
            return co.rowsum_cuda_backup(g_v)

    def transition_mtx_ln_blocks_all(self, step_nb):
        """
        constructs a transition matrix
        """

        if step_nb == len(self.hours_seq) - 1:  # last step, no transition
            if not self.cuda:
                return np.zeros((self.lattice_size, self.lattice_size), dtype=self.dtype_used)

            return gpa.zeros((self.lattice_size, self.lattice_size), dtype=self.dtype_used)

        # not at the last stop
        F_next_v = self.lattice_seq[step_nb]  # next lattice
        F_curr_v = self.lattice_seq[step_nb+1]  # curr. lattice

        if not self.cuda:
            P_m = np.empty((self.lattice_size, self.lattice_size))
            P_m_tmp = np.zeros(self.lattice_size + 1)  # tmp. mtx for taking differences

        else:
            P_m = gpa.empty((self.lattice_size, self.lattice_size), dtype=self.dtype_used)
            P_m_tmp = gpa.zeros(self.lattice_size + 1, dtype=self.dtype_used)

        for (F_curr_idx, F_curr) in enumerate(F_curr_v):
            P_m[F_curr_idx, :] = self.tm_ln_blocks_sg_fast(F_next_v.reshape((self.lattice_size, 1)),
                                                           F_curr,
                                                           self.market_seq[step_nb+1]["fwd"],
                                                           self.market_seq[step_nb]["fwd"],
                                                           self.market_seq[step_nb+1]["sigma_C"],
                                                           self.market_seq[step_nb]["sigma_C"],
                                                           0.9,  # rho WRONG WRONG WRONG
                                                           np.sum(self.t_diff_seq[:step_nb+2]),
                                                           self.t_diff_seq[step_nb+2])  # WRONG WRONG # .flatten()
            # if np.abs(P_m[F_curr_idx, -1]) > 1e-2:
            #     P_m_tmp[1:] = P_m[F_curr_idx, :] / P_m[F_curr_idx, -1]  # normalization - CHECK CHECK
            # P_m[F_curr_idx, :] = np.diff(P_m_tmp)

        logger.debug("Finished generating matrix = ", np.sum(P_m, axis = 1))
        return P_m

    def transit_val(self, P_m, H_m, G_m):
        """
        transition value of tolling:
          P_m ... transition matrix (or tensor)
          H_m ... next value of tolling
          G_m ... running profit
        """
        res_m = self.zero_pp()  # this is set by itself to cuda or no cuda

        if self.fuel_ind:
            s = P_m.shape[0]
            for F_1_ind in range(s):
                for F_2_ind in range(s):
                    res_m[F_1_ind, F_2_ind] = np.sum(P_m[F_1_ind, F_2_ind, :, :] * H_m) + G_m[F_1_ind, F_2_ind]
        else:
            if not self.cuda:
                dotp = np.dot
            else:  # matrix times a vector multiply
                # dotp = co.matmul_new
                dotp = cublas.cublasSgemv
            res_m = dotp(P_m, H_m) + G_m

        return res_m

    def construct_hours(self):
        """
        construct the hours from the blocks structure
        """
        days = np.array([0.])
        hours_seq = []
        market_seq = []
        lattice_seq = []
        for day in range(30):
            day_week = np.mod(day, 7)
            hours_market_for_day_week = [(hp, mp, lb) for (hp, dp, mp, lb) in
                                         zip(self.blocks.hours_partition, self.blocks.days_partition,
                                             self.blocks.market, self.lattice_blocks) if day_week in dp][0]
            hours_for_day_week, market_for_day_week, lattice_for_day_week = hours_market_for_day_week
            # market_for_day_week = [mp for (hp, mp) in hours_market_for_day_week]
            days = np.append(days, days[-1] + np.cumsum(hours_for_day_week) / (24. * 365.))
            hours_seq.extend(hours_for_day_week)
            market_seq.extend(market_for_day_week)
            lattice_seq.extend(lattice_for_day_week)

        self.nb_steps = len(hours_seq)  # number of steps for the lattice
        days_diff = np.empty(len(days))
        days_diff[0] = 0.
        days_diff[1:] = np.diff(days)

        return { 'blocks_tdiff': days_diff
               , 'hours_seq'   : hours_seq
               , 'market_seq'  : market_seq
               , 'lattice_seq' : lattice_seq}

    def running_profit(self, blocks_nb):
        """ Running profit for block nb_sims block_nb.

        :param blocks_nb: number of the block for which the running profit is computed.
        """

        # fixed tolling  TODO: gas tolling TO CORRECT TO CORRECT TO CORRECT
        profit_cap_indep = (self.lattice_seq[blocks_nb] - self.blocks.K) * self.hours_seq[blocks_nb]

        return profit_cap_indep * self.max_cap, profit_cap_indep * self.min_cap

    def plot_dispatch2(self, image):
        """
        image is a matrix of elts between 0,1
        """
        fig, ax = plt.subplots()
        nrows, ncols = image.shape

        # image = np.random.uniform(size=(10, 10))
        ax.imshow(image, cmap=plt.cm.hot, interpolation='nearest')
        ax.set_title('dispatch intensity plot')

        # Move left and bottom spines outward by 10 points
        ax.spines['left'].set_position(('outward', ncols))
        ax.spines['bottom'].set_position(('outward', nrows))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.show()

    def step(self, Ut_curr, Dt_curr, block_nb, start_nb=10):
        """
        Ut_curr ... current uptime of the PP
        Dt_curr ... current downtime of the PP
        block_nb ... which block nb_sims in the month are we currently at
        start_nb ... number of startups of the PP in that month NOT IMPLEMENTED YET
        """
        profit_max, profit_min = self.running_profit(block_nb)

        if not self.P_m_seq.has_key(block_nb):
            self.P_m_seq[block_nb] = self.transition_mtx_ln_blocks_all(block_nb)

        Pm = self.P_m_seq[block_nb]  # current transition matrix

        if Ut_curr == self.MUT:  # minimum uptime reached; options: ramp up/down, switch off
            self.work_pp_curr["max"][Ut_curr], self.work_pp_curr_ind[block_nb]["max"][Ut_curr] = maximum_3(
                        self.transit_val(Pm, self.idle_pp_next[0], profit_max - self.fixed_sd_cost),
                        self.transit_val(Pm, self.work_pp_next["min"][self.MUT], profit_max) - self.ramp_dn_cost,
                        self.transit_val(Pm, self.work_pp_next["max"][self.MUT], profit_max),
                        keep_dec=self.keep_dec, ci=self.cuda)
            self.work_pp_curr["min"][Ut_curr], self.work_pp_curr_ind[block_nb]["min"][Ut_curr] = maximum_3(
                        self.transit_val(Pm, self.idle_pp_next[0], profit_min - self.fixed_sd_cost),
                        self.transit_val(Pm, self.work_pp_next["min"][self.MUT], profit_min),
                        self.transit_val(Pm, self.work_pp_next["max"][self.MUT], profit_min) - self.ramp_up_cost,
                        keep_dec=self.keep_dec, ci=self.cuda)
        else:  # MUT not reached; we can not switch off, just ramp up/down
            self.work_pp_curr["max"][Ut_curr], self.work_pp_curr_ind[block_nb]["max"][Ut_curr] = maximum_2(
                    self.transit_val(Pm, self.work_pp_next["min"][Ut_curr + 1], profit_max) - self.ramp_dn_cost,
                    self.transit_val(Pm, self.work_pp_next["max"][Ut_curr + 1], profit_max),
                    keep_dec=self.keep_dec, ci=self.cuda)
            self.work_pp_curr_ind[block_nb]["max"][Ut_curr].fill(True)  # + 1 because state 0 is idle
            self.work_pp_curr["min"][Ut_curr], self.work_pp_curr_ind[block_nb]["min"][Ut_curr] = maximum_2(
                self.transit_val(Pm, self.work_pp_next["min"][Ut_curr + 1], profit_min),
                self.transit_val(Pm, self.work_pp_next["max"][Ut_curr + 1], profit_min) - self.ramp_up_cost,
                keep_dec=self.keep_dec, ci=self.cuda)
            self.work_pp_curr_ind[block_nb]["min"][Ut_curr].fill(True)

        if Dt_curr == self.MDT:  # min. downtime reached; can be restarted
            self.idle_pp_curr[Dt_curr], self.idle_pp_curr_ind[block_nb][Dt_curr] = maximum_3(
                        self.transit_val(Pm, self.idle_pp_next[self.MDT], self.zero_matrix),  # continuing idle
                        self.transit_val(Pm, self.work_pp_next["min"][0], - self.fixed_startup_cost),
                        self.transit_val(Pm, self.work_pp_next["max"][0], - self.fixed_startup_cost),  # restart to max
                        keep_dec=self.keep_dec, ci=self.cuda)

        else:  # can not be restarted, just run idle
            self.idle_pp_curr[Dt_curr] = self.transit_val(Pm, self.idle_pp_next[Dt_curr + 1], self.zero_matrix)
            self.idle_pp_curr_ind[block_nb][Dt_curr] = np.zeros(self.lattice_size)  # 0 is the IDLE indicator

    # all steps for 1 time step
    def all_one_steps(self, block_nb):
        self.step(0, 0, block_nb=block_nb)
        for Ut in range(1, self.MUT + 1):
            self.step(Ut, 0, block_nb=block_nb)
        for Dt in range(1, self.MDT + 1):
            self.step(0, Dt, block_nb=block_nb)

    def decisions_from_idle(self):
        """
        constructs the decision matrix from IDLE, for every blockl; and for every block a
        """
        state_mtx = np.zeros((self.lattice_size, 60))
        Dt_curr = np.zeros(self.lattice_size)  # current downtime
        Ut_curr = np.zeros(self.lattice_size)  # current downtime
        max_curr = np.zeros(self.lattice_size)  # current downtime
        min_curr = np.zeros(self.lattice_size)  # current downtime
        curr_dec = self.idle_pp_curr_ind[0][0]  # start from IDLE
        # curr_state = state_mtx[:,0]
        for block in range(1, 60):
            # downtime, uptime evolution
            shut_dec = curr_dec == 0
            work_dec = curr_dec != 0
            min_dec = curr_dec == 1
            max_dec = curr_dec == 2
            Dt_curr[shut_dec] = np.minimum(Dt_curr[shut_dec] + 1, self.MDT)
            Dt_curr[work_dec] = 0
            Ut_curr[work_dec] = np.minimum(Ut_curr[work_dec] + 1, self.MUT)
            Ut_curr[shut_dec] = 0

            curr_dec = np.zeros(self.lattice_size)
            for lat_idx in range(self.lattice_size):
                # curr_dec[Dt_curr > 0] = self.idle_pp_curr_ind[block][Dt_curr][Dt_curr>0]
                if Dt_curr[lat_idx] > 0:  # downtime scenario
                    curr_dec[lat_idx] = self.idle_pp_curr_ind[block][int(Dt_curr[lat_idx])][lat_idx]
                else:  # uptime scenario
                    if Ut_curr[lat_idx] > 0 and min_dec[lat_idx] == True:
                        curr_dec[lat_idx] = self.work_pp_curr_ind[block]["min"][int(Ut_curr[lat_idx])][lat_idx]
                    else:  # max is true
                        curr_dec[lat_idx] = self.work_pp_curr_ind[block]["max"][int(Ut_curr[lat_idx])][lat_idx]

            state_mtx[:, block] = curr_dec
        return state_mtx

    def overwrite_next_w_curr(self):
        """ Generates work_pp_next, idle_pp_next from the current.
        """

        for work_pp_ind in range(self.MUT+1):
            self.work_pp_next["max"][work_pp_ind] = self.work_pp_curr["max"][work_pp_ind]
            self.work_pp_next["min"][work_pp_ind] = self.work_pp_curr["min"][work_pp_ind]
        for idle_pp_ind in range(self.MDT+1):
            self.idle_pp_next[idle_pp_ind] = self.idle_pp_curr[idle_pp_ind]

    def multiple_steps(self, nb_blocks):
        for block_nb in range(nb_blocks - 1, -1, -1):  # walking over blocks
            logger.debug("Block ", block_nb, "of", nb_blocks)
            self.all_one_steps(block_nb)
            self.overwrite_next_w_curr()
            logger.debug("Step ", block_nb, " finished in.")

    def tolling_value(self):
        """
        compute the tolling value from partial tolls for the given month
        """

        self.multiple_steps(self.nb_steps)  # do the steps within the month

        # compute average over the values here self.work_pp_curr[0]
        F_0_init = self.market_seq[0]["fwd"]
        sigma_F_init = self.market_seq[0]["sigma_F"]
        T_m_init = self.blocks.Tm
        if not self.cuda:
            F_init = self.lattice_seq[0]
        else:
            F_init = self.lattice_seq[0].get()

        cdf_Tm = cdf_vec((np.log(F_init / F_0_init) + 0.5 * sigma_F_init ** 2 * T_m_init) /
                         (sigma_F_init * np.sqrt(T_m_init)))
        cdf_Tm_a = np.zeros(self.lattice_size + 1)
        cdf_Tm_a[1:] = cdf_Tm
        pdf_Tm = np.diff(cdf_Tm_a)
        if not self.cuda:
            results_max = self.work_pp_curr["max"][0]
            results_min = self.work_pp_curr["min"][0]
            results_idle = self.idle_pp_curr[0]
        else:
            results_max = self.work_pp_curr["max"][0].get()
            results_min = self.work_pp_curr["min"][0].get()
            results_idle = self.idle_pp_curr[0].get()

        best_strategy = np.maximum(np.maximum(results_min - self.fixed_startup_cost,
                                              results_max - self.fixed_startup_cost),
                                   results_idle)

        return np.dot(pdf_Tm, best_strategy)  # works faster than np.sum


class TollingModelLatticeAll:
    """ Summing over all the months in the previous model

    """

    def __init__( self
                , params
                , blocks
                , nb_months
                , keep_dec = False
                , mp_ind   = False
                , cuda_ind = False ):
        """

        :param params: same format as for TollingModelLattice
        :param market: market_v.Fv = vector of forward prices
                   market_v.sigma_Fv ... vec. of vols
                   market_v.sigma_Cv ... vec. of cash vols
        blocks ... structure of the blocks
        :param keep_dec: whether the model keeps decision
        :param mp_ind: multiprocessing indicator
        :param cuda_ind: cuda indicator
        """

        self.params = params
        self.nb_months = nb_months
        self.blocks = blocks  # list of block objects
        self.total_val = 0.
        self.keep_dec = keep_dec
        self.mp_ind = mp_ind
        self.cuda_ind = cuda_ind

    def compute_val_one_month(self, m):
        tm_curr = TollingModelLattice(self.params, self.blocks[m],
                                      keep_dec=self.keep_dec,
                                      cuda_ind=self.cuda_ind)
        tm_curr_val = tm_curr.tolling_value()
        return tm_curr_val

    def compute_val(self, mp_ind=False):
        """
        computes the value of the tolling
        :param mp_ind: whether to do this model in parallel
        """
        self.total_val = 0.

        if not mp_ind:
            for m in range(self.nb_months):
                tm_curr_val = self.compute_val_one_month(m)
                self.total_val += tm_curr_val
            # dec_from_idle_mtx = tm_curr.decisions_from_idle()
            # if display:
            #    tm_curr.plot_dispatch2((np.flipud(dec_from_idle_mtx) + 1.)/3.)
        else:
            nb_cores = mp.cpu_count()
            pool = mp.Pool(processes=nb_cores)
            C = pool.map(compute_val_one_month_wrap,
                         zip([self] * self.nb_months, range(self.nb_months)))
            pool.close()
            return C

        return self.total_val  # , dec_from_idle_mtx
