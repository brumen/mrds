# front office tolling model 
# see the front office doc. for other things 

# File defines:
import config 

import numpy as np
from typing import Dict

if config.CUDA_PRESENT:
    import pycuda.autoinit
    import pycuda.curandom
    import pycuda.gpuarray as gpa


class TollingModelLSM:
    """
    backward induction algorithm on the lattice
    params: parameters, which contain 
       .cuda ... indicator whether CUDA is present or not 
    #   .F0 ... forward price
    #   .K  ... cost of running a PP for that month 
    #   .sigma_F ... forward vol
    #   .sigma_C ... cash vol 
    #   .SC ... fixed startup costs 
    # debug_ind ... indicator whether debug is desired

    """

    def __init__( self
                , nb_steps     : int
                , lattice_size : int
                , tolling_params : Dict ):
        """

        :param nb_steps: number of steps in the lattice
        :param lattice_size: size of the lattice
        :param tolling_params: tolling parameters for the month
        """

        self.nb_steps = nb_steps

        #self.tolling_fast = params.tolling_fast # using the fast tensor routine from tolling_fast.pyx
        #self.tolling_fast_mt = params.tolling_fast_mt # using raw multi-threading

        self.lattice_size = lattice_size
        self.minimum_downtime = tolling_params['MDT']
        self.minimum_uptime   = tolling_params['MUT']
        self.maxCap           = tolling_params['maxCap']
        self.minCap           = tolling_params['minCap']

        # initial values of power, fuel
        self.F0               = tolling_params['F0'] # forward price for that month
        self.K                = tolling_params['K']  # cost of dispatch
        self.sigma_F          = tolling_params['sigma_F']  # forward vol for the month
        self.sigma_C          = tolling_params['sigma_C']  # cash vol for the month

    @property
    def F_v(self):
        """ Forward price lattice.

        TODO: FIX THIS, THIS IS JUNK.
        """

        return self.F0 * np.exp(-1. + np.arange (np.double (self.lattice_size))/np.double (self.lattice_size) * 2.)

    def daily_sim(self, F_v, sigma_v, cash_v, T_v, nb_sims : int):
        """ Daily simulation.

        :param F_v:  monthly forwards
        # sigma_v ... monthly vols
        # cash_v ... cash vols
        # assumption ... 30 days in a month
        # T_v ... forwards for these times
        :param nb_sims: number of simulations
        # bunch of correlations, but let's leave that aside
        # output:
        #

        """

        all_sim = []

        for (F, sigma, sigma_cash, T) in zip(F_v, sigma_v, cash_v, T_v): # simulate the month_v 
            initial_v = np.random.lognormal (mean = np.log (F) - 0.5 * sigma**2 * T, sigma=sigma * np.sqrt(T), size = nb_sims)
            
            day_incr = np.concatenate(( [np.log(initial_v)]
                                      , np.log(np.random.lognormal(mean  = - 0.5 * sigma_cash**2 * (1./365.)
                                                                   , sigma = sigma_cash * np.sqrt (1./365.)
                                                                   , size  = (29, nb_sims))))
                                      , axis=0 )
            
            all_sim.append(np.exp(np.cumsum(day_incr.transpose(), 1)))

        return all_sim

    # do daily simulations on the cuda device 
    def daily_sim_cuda (self, F_v, sigma_v, cash_v, T_v, nb):
        
        all_sim = []

        rng = pycuda.curandom.Sobol64RandomNumberGenerator()

        # cumsum kernel
        cumsum_string = open ("cumsum_cuda.cu").read()
        cumsum_mod = config.SourceModule(cumsum_string % {"mat_cols": 29})
        cumsum_chosen = "cumsum_cuda"
        cumsum_prod = cumsum_mod.get_function (cumsum_chosen) # cumsum 
        block_sel = (1, 1, 1)
        grid_sel = (nb, 1)

        for (F, sigma, sigma_cash, T) in zip(F_v, sigma_v, cash_v, T_v): # simulate the month_v 
            rns = gpa.zeros(nb, np.float32)
            rng.fill_normal(rns)
            #mean1 = log (F) - 0.5 * sigma**2 * T
            #std1 = sigma * sqrt(T)
            #initial_v = float(mean1) + float(std1) * rns

            mean2 = - 0.5 * sigma_cash**2 * (1./365.)
            std2 = sigma_cash * np.sqrt (1./365.)
            rns2 = gpa.zeros((29,nb), np.float32)
            rng.fill_normal(rns2)
            day_incr = float(mean2) + float(std2) * rns2

            # CORRECT, CORRECT, the KERNEL DOES NOT WORK 
            cumsum_prod ( day_incr, block=block_sel, grid=grid_sel)  # cumsum on day_incr

            # day_curr = initial_v
            # day_incr = concatenate(([log(initial_v)], log (numpy.random.lognormal (mean = - 0.5 * sigma_cash**2 * (1./365.), sigma=sigma_cash * sqrt (1./365.), size = (29, nb_sims)) ) ), axis=0)
            
            #all_sim.append (exp (cumsum(day_incr.transpose(),1)) )
            all_sim.append (day_incr)

        return all_sim

    def _daily_profits(self, power_pr, fuel_pr, hr):
        return np.maximum(power_pr - hr * fuel_pr, 0.)

    # LSM algorithm on the simulations  
    def _daily_lsm(self, daily_sims_power, daily_sims_fuel):
        nb_months = len (daily_sims_power)
        nb_sims = daily_sims_power[0].shape[0]

        # for every month 
        V_u_all = np.zeros ((nb_sims, nb_months))
        V_d_all = np.zeros ((nb_sims, nb_months))
        for (month_nb, power_sim, fuel_sim) in zip (arange(nb_months), daily_sims_power, daily_sims_fuel):
            V_u = np.zeros (nb_sims)
            V_d = np.zeros (nb_sims)
            for day_sim_nb in np.arange(29,-1,-1): # daily sims from the back
                # regression 
                power_curr = power_sim[:,day_sim_nb]
                fuel_curr = fuel_sim[:,day_sim_nb]
                curr_profit = self._daily_profits(power_curr, fuel_curr, 7.)

                V_u_u = curr_profit + V_u 
                V_u_d = V_u
                V_d_u = curr_profit + V_u - 1. # 1 is cost 
                V_d_d = V_d 
                
                # switch option 
                sw_u = V_u_d - V_u_u 
                sw_d = V_d_u - V_d_d
                
                # regression 
                regs = np.array([np.ones(nb_sims), power_curr, fuel_curr])
                regs = regs.transpose()
                B_d = regress ( sw_d, regs )
                B_u = regress ( sw_u, regs )

                # switch decision 
                sw_dec_u = (B_u[0] + B_u[1] * power_curr + B_u[2] * fuel_curr > 0)
                sw_dec_d = (B_d[0] + B_d[1] * power_curr + B_d[2] * fuel_curr > 0)

                # current values updated 
                V_d = sw_dec_d * V_d_u + (1-sw_dec_d) * V_d_d 
                V_u = sw_dec_u * V_u_d + (1-sw_dec_u) * V_u_u
                
            V_d_all[:, month_nb] = V_d
            V_u_all[:, month_nb] = V_u

        return (B_d, B_u, V_d_all, V_u_all)


class TollingModelLSMCuda(TollingModelLSM):
    """ CUDA version of tolling model on lattice.
    """

    def _daily_profits(self, power_pr, fuel_pr, hr):
        """ Daily profits used on the device

        """
        return gpa.maximum(power_pr - hr * fuel_pr, gpa.zeros(len(power_pr), float) )

    def _daily_lsm(self, daily_sims_power_d, daily_sims_fuel_d):
        """ LSM algorithm on the simulations CUDA version

        """
        nb_months = len(daily_sims_power_d)
        nb_sims = daily_sims_power_d[0].shape[0]

        # for every month
        V_u_all = gpa.np.zeros((nb_sims, nb_months), float)
        V_d_all = gpa.zeros((nb_sims, nb_months), float)

        get_col_string = open("cumsum_cuda.cu").read()
        get_col_mod = config.SourceModule(get_col_string % {"mat_cols": 29})  # nb_days
        get_col_prod = get_col_mod.get_function("get_col")  # cumsum
        block_sel = (1, 1, 1)
        grid_sel = (nb_sims, 1)

        for (month_nb, power_sim, fuel_sim) in zip(arange(nb_months), daily_sims_power_d, daily_sims_fuel_d):
            V_u = gpa.zeros(nb_sims, float)
            V_d = gpa.zeros(nb_sims, float)
            for day_sim_nb in arange(29, -1, -1):  # daily sims from the back
                #
                # power_curr = power_sim[:,day_sim_nb]
                # fuel_curr = fuel_sim[:,day_sim_nb]
                power_curr = gpa.zeros(nb_sims, float)
                fuel_curr = gpa.zeros(nb_sims, float)

                get_col_prod(power_sim, gpa.to_gpu(array([day_sim_nb])).astype(float32),
                             gpa.to_gpu(array([nb_sims])).astype(float32), power_curr, block=block_sel, grid=grid_sel)
                get_col_prod(fuel_sim, gpa.to_gpu(array([day_sim_nb])).astype(float32),
                             gpa.to_gpu(array([nb_sims])).astype(float32),
                             fuel_curr, block=block_sel, grid=grid_sel)

                curr_profit = self._daily_profits(power_curr, fuel_curr, 7.)

                V_u_u = curr_profit + V_u
                V_u_d = V_u
                V_d_u = curr_profit + V_u - 1.  # 1 is cost
                V_d_d = V_d

                # switch option
                sw_u = V_u_d - V_u_u
                sw_d = V_d_u - V_d_d

                # sgesv - magma routine for solving linear equations

                # regression
                # regs = gpa([ones(nb_sims), power_curr, fuel_curr])
                # regs = regs.transpose()
                # B_d = regress_gpu ( sw_d, regs )
                # B_u = regress_gpu ( sw_u, regs )
                B_d = 0
                B_u = 0

                # switch decision
                # sw_dec_u = (B_u[0] + B_u[1] * power_curr + B_u[2] * fuel_curr > 0)
                # sw_dec_d = (B_d[0] + B_d[1] * power_curr + B_d[2] * fuel_curr > 0)

                # current values updated
                # V_d = sw_dec_d * V_d_u + (1-sw_dec_d) * V_d_d
                # V_u = sw_dec_u * V_u_d + (1-sw_dec_u) * V_u_u

            # V_d_all[:, month_nb] = V_d
            # V_u_all[:, month_nb] = V_u

        return (B_d, B_u, V_d_all, V_u_all)


# regression - x has to be a matrix 
def regress(y, x):
    nb_vars = x.shape[1]
    if nb_vars == 1:
        return sum(x * y)/sum (x * x)
    else:
        return np.solve(np.dot(x.transpose(), x), np.dot (x.transpose(), y))


# regression - x has to be a matrix, y,x are both gpu arrays
def regress_gpu(y, x):
    nb_vars = x.shape[1]
    if nb_vars == 1:
        return gpa.sum(x * y,dtype=float)/gpa.sum (x**2,dtype=float)
    else: # using 
        return solve(dot(x.transpose(), x)  , dot (x.transpose(), y))

