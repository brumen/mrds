# front office tolling model 
# see the front office doc. for other things 

# File defines:
import config

import numpy as np
import scipy.integrate
from numpy import *
import time
import multiprocessing
import ctypes
import numexpr as ne

# my modules
import tolling_fast
import lattice
#from pricers import cdf_vec 
from cond_prob import transition_mtx_ln_blocks_fast, transition_mtx_ln_blocks_fast_internal
import sg # sparse grids for fast integration

if config.CUDA_PRESENT:
    import config.gpuarray as gpa


# temporary fix, this function is in pricers
def cdf_vec(x):
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429

    L = abs(array(x))

    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / sqrt(2 * pi) * exp(-L * L / 2) * (a1 * K + a2 * K * K + a3 * K ** 3 + a4 * K ** 4 + a5 * K ** 5)

    return w * (x >= 0.) + (1.0 - w) * (x < 0.)

def cdf_vec_ne(x):
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    pi_local = 3.141592653589793

    L = ne.evaluate("abs(x)")

    K = ne.evaluate("1.0 / (1.0 + 0.2316419 * L)")
    w = ne.evaluate("1.0 - 1.0 / sqrt(2 * pi_local) * exp(-L * L / 2) * (a1 * K + a2 * K * K + a3 * K ** 3 + a4 * K ** 4 + a5 * K ** 5)")

    return ne.evaluate("w * (x >= 0.) + (1.0 - w) * (x < 0.)")

def pdf_vec(x):
    return np.exp(-x**2/2.)/np.sqrt(2. * np.pi)


# multi-threading version of tensor product
tens_fast_mt_raw = ctypes.CDLL("/home/brumen/workspace/mrds/tp.so").tensor_prod_2


def tens_fast_mt(P_m, H_m, G_m, res_m):
    tens_fast_mt_raw(ctypes.py_object(P_m),
                     ctypes.py_object(H_m),
                     ctypes.py_object(G_m),
                     ctypes.py_object(res_m))


# structure for tolling parameters
class tolling_params():
    pass


# wrapper for the skew MRD model calibration function
def transition_wrap(arg, **kwarg):
    return tolling_model_MRD.transition_mtx_optim(*arg, **kwarg)


def step_wrap(arg, **kwarg):
    return tolling_model_MRD.step(*arg, **kwarg)


class tolling_model_lattice():
    """
    # backward induction algorithm on the lattice
    # params: parameters, which contain 
    #   .cuda ... indicator whether CUDA is present or not 
    #   .F0 ... forward price
    #   .K  ... cost of running a PP for that month 
    #   .sigma_F ... forward vol
    #   .sigma_C ... cash vol 
    #   .SC ... fixed startup costs 
    # debug_ind ... indicator whether debug is desired

    """

    def __init__(self, params, blocks, debug_ind=False):
        #self.comp_ind = params.comp_ind # computational indicator: 0 ... slow CPU, 1 ... fast CPU, 2 ... MT CPU, 3 ... CUDA 
        self.cuda = params.cuda
        self.tolling_fast = params.tolling_fast # using the fast tensor routine from tolling_fast.pyx
        self.tolling_fast_mt = params.tolling_fast_mt # using raw multi-threading 

        self.params = params ## THIS CAN PERHAPS BE REMOVED LATER
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



        # blocks
        self.blocks = blocks
        # construction of lattice blocks for each market, should eventually replace above
        self.lattice_blocks_by_name = {}
        self.market_by_name = {}
        for dp in self.blocks.market:
            for m in dp:
                market_name = m["name"]
                if not self.lattice_blocks_by_name.has_key(market_name):
                    self.lattice_blocks_by_name[market_name] = lattice.lattice_ln(m["fwd"],
                                                                                  m["sigma_F"],
                                                                                  m["sigma_C"],
                                                                                  self.lattice_size).lattice
                if not self.market_by_name.has_key(market_name):
                    self.market_by_name[market_name] = m

        # lattice per blocks; done so that it uses pointers
        self.lattice_blocks = [ [self.lattice_blocks_by_name[m["name"]] for m in dp]
                                for dp in self.blocks.market ]

        self.hours_seq = self.construct_hours()["hours_seq"]
        self.t_diff_seq = self.construct_hours()["blocks_tdiff"]
        self.market_seq = self.construct_hours()["market_seq"] # sequence of market moves
        self.lattice_seq = self.construct_hours()["lattice_seq"]

        self.P_m_seq = {} # sequence of transition matrices, currently empty, filled in as the program progresses


        self.fuel_ind = False # whether fuel is present
        self.debug_ind = debug_ind


        # profit matrix/vector
        self.zero_matrix = self.zero_pp()
        if self.cuda:
            self.zero_matrix_d = gpa.to_gpu(self.zero_pp()).astype(float32)

        # tmp. storage 
        if not self.cuda:
            self.work_curr_tmp = {"max": self.zero_pp(), "min": self.zero_pp()} # work between max and min capacity
            self.idle_curr_tmp = self.zero_pp()
        else:
            self.work_curr_tmp = gpa.to_gpu(self.zero_pp()).astype(float32)
            self.idle_curr_tmp = gpa.to_gpu(self.zero_pp()).astype(float32)

        # the next power plant (pp) working condition 
        self.work_pp_next = {"max": [], "min": []} # max for max capacity, min for min cap.; only these two states cons.
        self.work_pp_curr = {"max": [], "min": []}
        self.idle_pp_next = []
        self.idle_pp_curr = []
        for Ut in range(self.MUT + 1):
            if not self.cuda:
                self.work_pp_next["max"].append(self.zero_pp())
                self.work_pp_curr["max"].append(self.zero_pp())
                self.work_pp_next["min"].append(self.zero_pp())
                self.work_pp_curr["min"].append(self.zero_pp())
            else:
                self.work_pp_next["max"].append(gpa.to_gpu(self.zero_pp()).astype(float32))
                self.work_pp_curr["max"].append(gpa.to_gpu(self.zero_pp()).astype(float32))
                self.work_pp_next["min"].append(gpa.to_gpu(self.zero_pp()).astype(float32))
                self.work_pp_curr["min"].append(gpa.to_gpu(self.zero_pp()).astype(float32))

        for Dt in range(self.MDT + 1):
            if not self.cuda:
                self.idle_pp_next.append(self.zero_pp())
                self.idle_pp_curr.append(self.zero_pp())
            else:
                self.idle_pp_next.append(gpa.to_gpu(self.zero_pp()).astype(float32))
                self.idle_pp_curr.append(gpa.to_gpu(self.zero_pp()).astype(float32))


    # construction of the zero matrix
    def zero_pp(self):
        if self.fuel_ind:
            return np.zeros((self.lattice_size, self.lattice_size)) # matrix
        else:
            return np.zeros(self.lattice_size) # vector





    def transition_mtx_ln_blocks_sg_fast_helper(self, v, p_dash, op, F_P, F_OP, sigma_P, sigma_OP, rho, t, delta_t=1./365.,
                                                sg_level = 15):

        z_op = (log(op/F_OP) + 0.5 * sigma_OP**2 * t) / (sigma_OP * sqrt(t) )
        # p is vector
        # z_p = (np.log(p/F_P) + 0.5 * sigma_P**2 * t) / (sigma_P * np.sqrt(t) )
        # v = (z_p - rho * z_op) / sqrt(1. - rho**2)

        d1 = (log(p_dash/F_P) + 0.5 * sigma_P**2 * (t+delta_t) - sqrt(1. - rho**2) * sigma_P * sqrt(t) * v \
              - rho * z_op * sigma_P * sqrt(t) ) / (sigma_P * sqrt(delta_t))

        return cdf_vec_ne(d1)



    def transition_mtx_ln_blocks_sg_fast(self, p_dash, op, F_P, F_OP, sigma_P, sigma_OP, rho, t, delta_t=1./365.,
                                         sg_level = 15):
        """
        sparse grid integration without functions
          level ... sg level, how fine the sparse grid gets
          xs ... abscisses over which the integration is done
        """
        weights = np.array(sg.sg_w(1, sg_level)) # row vector
        xs = np.array(sg.sg_p(1, sg_level)).flatten() # row vector

        sqrt2 = np.sqrt(2.)
        sqrt_pi = np.sqrt(np.pi)


        # p_dash has to be column vector
        g_v = self.transition_mtx_ln_blocks_sg_fast_helper((xs * sqrt2),
                                                           p_dash, op, F_P, F_OP, sigma_P, sigma_OP, rho, t, delta_t,
                                                           sg_level) / sqrt_pi # g = lambda x: f(x * sqrt2) / (sqrt_pi)**D

        # vals = np.array(map (g, map(lambda x: np.array(x), sg_p(D, l, one_d_discret)))).flatten()
        return np.sum(weights * g_v, axis = 1)



    def transition_mtx_ln_blocks_all(self, step_nb):
        """
        constructs a transition matrix
        """
        if step_nb == len (self.hours_seq) -1: # last step, no transition
            return np.zeros ((self.lattice_size, self.lattice_size))
        else:
            F_next_v = self.lattice_seq[step_nb] # next lattice
            F_curr_v = self.lattice_seq[step_nb+1] # curr. lattice
            P_m = np.zeros ((self.lattice_size, self.lattice_size))
            P_m_tmp = np.zeros (self.lattice_size + 1) # tmp. mtx for taking differences
            for (F_curr_idx, F_curr) in enumerate(F_curr_v):
                P_m[F_curr_idx, :] = self.transition_mtx_ln_blocks_sg_fast(F_next_v.reshape((self.lattice_size,1)),
                                                                           F_curr,
                                                                           self.market_seq[step_nb+1]["fwd"],
                                                                           self.market_seq[step_nb]["fwd"],
                                                                           self.market_seq[step_nb+1]["sigma_C"],
                                                                           self.market_seq[step_nb]["sigma_C"],
                                                                           0.9, # rho WRONG WRONG WRONG
                                                                           np.sum(self.t_diff_seq[:step_nb+2]),
                                                                           self.t_diff_seq[step_nb+2] # WRONG WRONG WRONG
                                                                           ).flatten()
                if np.abs(P_m[F_curr_idx,-1])> 1e-2:
                    P_m_tmp[1:] = P_m[F_curr_idx,:] / P_m[F_curr_idx,-1] # normalization - NOT SURE IF RIGHT??????????
                P_m[F_curr_idx,:] = np.diff(P_m_tmp)

            #if self.debug_ind:
            #print "Finished generating matrix = ", np.sum(P_m, axis = 1)
            #    print "Finished generating matrix = ", np.sum(P_m[-1,:])

            return P_m

    # multi-threading version of the transition matrix
    def transition_mtx_multi(self):
        nb_cores = multiprocessing.cpu_count()
        print "Using ", nb_cores, " cores."
        pool = multiprocessing.Pool(processes=nb_cores)
        print "Starting worker threads"
        C = pool.map(transition_wrap, zip([self] * len(self.F_1_v),
                                          xrange(len(self.F_1_v))))

        return C


    def transit_val(self, P_m, H_m, G_m):
        """
        transition value of tolling:
          P_m ... transition matrix (or tensor)
          H_m ... next value of tolling
          G_m ... running profit 
        """
        res_m = self.zero_pp()

        if self.fuel_ind:
            s = P_m.shape[0]
            for F_1_ind in range(s):
                for F_2_ind in range(s):
                    res_m[F_1_ind, F_2_ind] = np.sum(P_m[F_1_ind, F_2_ind, :, :] * H_m) + G_m[F_1_ind, F_2_ind]

        else:
            res_m = np.dot(P_m, H_m) + G_m

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
            #market_for_day_week = [mp for (hp, mp) in hours_market_for_day_week]
            days = np.append(days, days[-1] + np.cumsum(hours_for_day_week) / 24. / 365.)
            hours_seq.extend(hours_for_day_week)
            market_seq.extend(market_for_day_week)
            lattice_seq.extend(lattice_for_day_week)

        self.nb_steps = len (hours_seq) # number of steps for the lattice

        days_diff = np.zeros(len(days))
        days_diff[1:] = np.diff(days)

        return {"blocks_tdiff": days_diff, "hours_seq": hours_seq,
                "market_seq": market_seq, "lattice_seq": lattice_seq}


    def running_profit(self, blocks_nb):
        """
        running profit function
        """
        if not self.fuel_ind: # fixed tolling
            profit_cap_indep = (self.lattice_seq[blocks_nb] - self.blocks.K) * self.hours_seq[blocks_nb]

        else: # gas tolling TO CORRECT TO CORRECT TO CORRECT
            profit_cap_indep = (self.lattice_seq[blocks_nb] - self.blocks.K) * self.hours_seq[blocks_nb]

        return (profit_cap_indep * self.maxCap, profit_cap_indep * self.minCap)

    def maximum_2(self, x, y):
        """
        computes the maximum and the indices:
        0 for 1st, 1 for 2nd)
        """
        xy_max = np.maximum (x,y)
        xy_ind = (xy_max == y)

        return (xy_max, xy_ind)

    def maximum_3(x,y,z):
        """
        0 - for maximum in x
        1 - maximum in y
        2 - maximum in z
        """
        xy_max, xy_ind = self.maximum_2(x,y)
        xyz_max = np.maximum (xy_max, z)
        xyz_ind = (xyz_max == xy_max) * xy_ind + (xyz_max == z) * 2 * np.ones(len(x))

        return (xyz_max, xyz_ind)

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
        block_nb ... which block nb in the month are we currently at
        start_nb ... number of startups of the PP in that month NOT IMPLEMENTED YET
        """

        profit_max, profit_min = self.running_profit(block_nb)

        if not self.P_m_seq.has_key(block_nb):
            self.P_m_seq[block_nb] = self.transition_mtx_ln_blocks_all(block_nb)

        Pm = self.P_m_seq[block_nb] # current transition matrix


        if (Ut_curr == self.MUT): # minimum uptime reached; options: ramp up/down, switch off
            #self.work_pp_curr["max"][Ut_curr] = np.maximum(
            #        np.maximum(
            #            self.transit_val(Pm, self.work_pp_next["max"][self.MUT], profit_max),
            #            self.transit_val(Pm, self.work_pp_next["min"][self.MUT], profit_max) - self.rampDnCost),
            #        self.transit_val(Pm, self.idle_pp_next[0], profit_max - self.fixedShutdownCost)
            #)
            self.work_pp_curr["max"][Ut_curr], a1 = self.maximum_3(
                        self.transit_val(Pm, self.work_pp_next["max"][self.MUT], profit_max),
                        self.transit_val(Pm, self.work_pp_next["min"][self.MUT], profit_max) - self.rampDnCost,
                        self.transit_val(Pm, self.idle_pp_next[0], profit_max - self.fixedShutdownCost)
            )

            self.work_pp_curr["min"][Ut_curr] = np.maximum(
                    np.maximum(
                        self.transit_val(Pm, self.work_pp_next["min"][self.MUT], profit_min),
                        self.transit_val(Pm, self.work_pp_next["max"][self.MUT], profit_min) - self.rampUpCost),
                    self.transit_val(Pm, self.idle_pp_next[0], profit_min - self.fixedShutdownCost)
            )
        else: # MUT not reached; we can not switch off, just ramp up/down
            self.work_pp_curr["max"][Ut_curr] = np.maximum(
                    self.transit_val(Pm, self.work_pp_next["max"][Ut_curr + 1], profit_max),
                    self.transit_val(Pm, self.work_pp_next["min"][Ut_curr + 1], profit_max) - self.rampDnCost
            )
            self.work_pp_curr["min"][Ut_curr] = np.maximum(
                    self.transit_val(Pm, self.work_pp_next["min"][Ut_curr + 1], profit_min),
                    self.transit_val(Pm, self.work_pp_next["max"][Ut_curr + 1], profit_min) - self.rampUpCost
            )


        if (Dt_curr == self.MDT): # min. downtime reached; can be restarted
            self.idle_pp_curr[Dt_curr] = np.maximum(
                    np.maximum (
                        self.transit_val(Pm, self.idle_pp_next[self.MDT], self.zero_matrix), # continuing idle
                        self.transit_val(Pm, self.work_pp_next["max"][0], - self.fixedStartupCost) # restarting to max
                    ),
                self.transit_val(Pm, self.work_pp_next["min"][0], - self.fixedStartupCost)
            )

        else: # can not be restarted, just run idle
            self.idle_pp_curr[Dt_curr] = self.transit_val(Pm, self.idle_pp_next[Dt_curr + 1], self.zero_matrix)


    # all steps for 1 time step 
    def all_one_steps(self, block_nb):
        if not self.cuda:
            if self.tolling_fast: # fast version of tensor product
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

    # multithreading version of the above function
    def all_one_steps_mt(self):
        nb_cores = multiprocessing.cpu_count()
        print "Using ", nb_cores, " cores."
        pool = multiprocessing.Pool(processes=nb_cores)
        print "Starting multithread one step"
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
        for block_nb in range(nb_blocks - 1, -1, -1): # walking over blocks
            #if self.debug_ind:
            print "Block ", block_nb, "of", nb_blocks
            self.all_one_steps(block_nb)
            self.overwrite_next_w_curr()

            if self.debug_ind:
                print "Step ", block_nb, " finished in."


    def tolling_value(self):
        """
        compute the tolling value from partial tolls for the given month
        """

        self.multiple_steps(self.nb_steps) # do the steps within the month
        if not self.cuda:
            # compute average over the values here self.work_pp_curr[0]
            F_init = self.lattice_seq[0]
            F_0_init = self.market_seq[0]["fwd"]
            sigma_F_init = self.market_seq[0]["sigma_F"]
            T_m_init = self.blocks.Tm

            cdf_Tm = cdf_vec((np.log(F_init / F_0_init ) + 0.5 * sigma_F_init ** 2 * T_m_init ) /
                             (sigma_F_init * np.sqrt(T_m_init)))
            cdf_Tm_a = np.zeros(self.lattice_size + 1)
            cdf_Tm_a[1:] = cdf_Tm
            pdf_Tm = np.diff(cdf_Tm_a)
            best_strategy = np.maximum (np.maximum(self.work_pp_curr["min"][0] - self.fixedStartupCost,
                                                   self.work_pp_curr["max"][0] - self.fixedStartupCost),
                                        self.idle_pp_curr[0])
            return np.sum(pdf_Tm * best_strategy)

        else:
            curr_mtx = self.work_pp_curr[0].get()
            return curr_mtx[self.lattice_size / 2, self.lattice_size / 2]


class tolling_model_lattice_all():
    """
        summing over all the months in the previous model

    """

    def __init__(self, params, blocks, nb_months):
        """
        params ... same format as for tolling_model_lattice
        market ... market_v.Fv = vector of forward prices
                   market_v.sigma_Fv ... vec. of vols
                   market_v.sigma_Cv ... vec. of cash vols
        blocks ... structure of the blocks
        """

        self.params = params
        self.nb_months = nb_months
        self.blocks = blocks # list of block objects

    def compute_val(self):
        self.total_val = 0.
        for m in range(self.nb_months):
            tm_curr = tolling_model_lattice(self.params, self.blocks[m])
            tm_curr_val = tm_curr.tolling_value()
            #print "TV1 = ", tm_curr_val
            self.total_val += tm_curr_val
        return self.total_val

