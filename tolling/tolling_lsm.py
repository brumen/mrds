# front office tolling model 
# see the front office doc. for other things 

# File defines:
import config 

import numpy
from numpy import *
import scipy
import scipy.optimize

import time
import openopt 
import pycuda.curandom 
import multiprocessing
import ctypes 


import tolling_fast
from pricers import cdf_vec 
import config.gpuarray as gpa 


# structure for tolling parameters 
class tolling_params():
    pass


class tolling_model_lsm():
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

    def __init__ (self, params, debug_ind=False):
        #self.comp_ind = params.comp_ind # computational indicator: 0 ... slow CPU, 1 ... fast CPU, 2 ... MT CPU, 3 ... CUDA 
        self.cuda = params.cuda 
        self.nb_steps = params.nb_steps
        self.tolling_fast = params.tolling_fast # using the fast tensor routine from tolling_fast.pyx
        self.tolling_fast_mt = params.tolling_fast_mt # using raw multi-threading 
        self.params = params ## THIS CAN PERHAPS BE REMOVED LATER
        self.lattice_size = params.lattice_size
        self.minimum_downtime = params.MDT
        self.minimum_uptime = params.MUT
        self.maxCap = params.maxCap
        self.minCap = params.minCap
        self.debug_ind = debug_ind
        # initial values of power, fuel 
        self.F0 = params.F0 # forward price for that month 
        self.K = params.K  # cost of dispatch 
        self.sigma_F = params.sigma_F # forward vol for the month 
        self.sigma_C = params.sigma_C # cash vol for the month 
        # lattice construction 
        self.F_v = self.F0 * np.exp ( -1. + arange (double (params.lattice_size))/double (params.lattice_size) * 2. )



    # daily simulation 
    # F_v ... monthly forwards 
    # sigma_v ... monthly vols 
    # cash_v ... cash vols 
    # assumption ... 30 days in a month 
    # T_v ... forwards for these times 
    # nb ... number of simulations 
    # bunch of correlations, but let's leave that aside 
    # output: 
    #    
    def daily_sim (self, F_v, sigma_v, cash_v, T_v, nb):
        
        all_sim = []

        for (F, sigma, sigma_cash, T) in zip(F_v, sigma_v, cash_v, T_v): # simulate the month_v 
            initial_v = numpy.random.lognormal (mean = log (F) - 0.5 * sigma**2 * T, sigma=sigma * sqrt (T) , size = nb) 
            
            #day_curr = initial_v
            day_incr = concatenate(([log(initial_v)], log (numpy.random.lognormal (mean = - 0.5 * sigma_cash**2 * (1./365.), sigma=sigma_cash * sqrt (1./365.), size = (29, nb)) ) ), axis=0)
            
            all_sim.append (exp (cumsum(day_incr.transpose(),1)) )

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
        block_sel = (1,1,1)
        grid_sel = (nb,1)

        for (F, sigma, sigma_cash, T) in zip(F_v, sigma_v, cash_v, T_v): # simulate the month_v 
            rns = gpa.zeros (nb, float32)
            rng.fill_normal(rns)
            #mean1 = log (F) - 0.5 * sigma**2 * T
            #std1 = sigma * sqrt(T)
            #initial_v = float(mean1) + float(std1) * rns

            mean2 = - 0.5 * sigma_cash**2 * (1./365.)
            std2 = sigma_cash * sqrt (1./365.)
            rns2 = gpa.zeros((29,nb), float32)
            rng.fill_normal(rns2)
            day_incr = float(mean2) + float(std2) * rns2

            # CORRECT, CORRECT, the KERNEL DOES NOT WORK 
            cumsum_prod ( day_incr, block=block_sel, grid=grid_sel ) # cumsum on day_incr
            
            
            # day_curr = initial_v
            # day_incr = concatenate(([log(initial_v)], log (numpy.random.lognormal (mean = - 0.5 * sigma_cash**2 * (1./365.), sigma=sigma_cash * sqrt (1./365.), size = (29, nb)) ) ), axis=0)
            
            #all_sim.append (exp (cumsum(day_incr.transpose(),1)) )
            all_sim.append (day_incr)

        return all_sim



    # simplified daily profits
    def daily_profits(self, power_pr, fuel_pr, hr):
        return maximum (power_pr - hr * fuel_pr,0)

    # daily profits used on the device 
    def daily_profits_cuda(self, power_pr, fuel_pr, hr):
        return gpa.maximum (power_pr - hr * fuel_pr, gpa.zeros(len(power_pr), float) )


    # LSM algorithm on the simulations  
    def daily_lsm (self, daily_sims_power, daily_sims_fuel):
        nb_months = len (daily_sims_power)
        nb_sims = daily_sims_power[0].shape[0]

        # for every month 
        V_u_all = zeros ((nb_sims, nb_months))
        V_d_all = zeros ((nb_sims, nb_months))
        for (month_nb, power_sim, fuel_sim) in zip (arange(nb_months), daily_sims_power, daily_sims_fuel):
            V_u = zeros (nb_sims)
            V_d = zeros (nb_sims)
            for day_sim_nb in arange(29,-1,-1): # daily sims from the back 
                # regression 
                power_curr = power_sim[:,day_sim_nb]
                fuel_curr = fuel_sim[:,day_sim_nb]
                curr_profit = self.daily_profits(power_curr, fuel_curr, 7.)

                V_u_u = curr_profit + V_u 
                V_u_d = V_u
                V_d_u = curr_profit + V_u - 1. # 1 is cost 
                V_d_d = V_d 
                
                # switch option 
                sw_u = V_u_d - V_u_u 
                sw_d = V_d_u - V_d_d
                
                # regression 
                regs = array([ones(nb_sims), power_curr, fuel_curr])
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


    # LSM algorithm on the simulations CUDA version   
    def daily_lsm_cuda (self, daily_sims_power_d, daily_sims_fuel_d):
        nb_months = len (daily_sims_power_d)
        nb_sims = daily_sims_power_d[0].shape[0]
        
        # for every month 
        V_u_all = gpa.zeros ((nb_sims, nb_months), float)
        V_d_all = gpa.zeros ((nb_sims, nb_months), float)


        get_col_string = open ("cumsum_cuda.cu").read()
        get_col_mod = config.SourceModule(get_col_string % {"mat_cols": 29}) # nb_days
        get_col_prod = get_col_mod.get_function ("get_col") # cumsum 
        block_sel = (1,1,1)
        grid_sel = (nb_sims,1)

        for (month_nb, power_sim, fuel_sim) in zip (arange(nb_months), daily_sims_power_d, daily_sims_fuel_d):
            V_u = gpa.zeros (nb_sims, float)
            V_d = gpa.zeros (nb_sims, float)
            for day_sim_nb in arange(29,-1,-1): # daily sims from the back 
                # 
                # power_curr = power_sim[:,day_sim_nb]
                # fuel_curr = fuel_sim[:,day_sim_nb]
                power_curr = gpa.zeros (nb_sims, float)
                fuel_curr = gpa.zeros (nb_sims, float)

                get_col_prod (power_sim, gpa.to_gpu(array([day_sim_nb])).astype(float32), 
                              gpa.to_gpu(array([nb_sims])).astype(float32), power_curr, block=block_sel, grid=grid_sel)
                get_col_prod (fuel_sim, gpa.to_gpu(array([day_sim_nb])).astype(float32), 
                              gpa.to_gpu(array([nb_sims])).astype(float32), 
                              fuel_curr, block=block_sel, grid=grid_sel)

                curr_profit = self.daily_profits_cuda ( power_curr, fuel_curr, 7.)
                
                V_u_u = curr_profit + V_u 
                V_u_d = V_u
                V_d_u = curr_profit + V_u - 1. # 1 is cost 
                V_d_d = V_d 
                
                # switch option 
                sw_u = V_u_d - V_u_u
                sw_d = V_d_u - V_d_d

                # sgesv - magma routine for solving linear equations
                
                # regression 
                #regs = gpa([ones(nb_sims), power_curr, fuel_curr])
                #regs = regs.transpose()
                #B_d = regress_gpu ( sw_d, regs )
                #B_u = regress_gpu ( sw_u, regs )
                B_d = 0
                B_u = 0

                # switch decision 
                #sw_dec_u = (B_u[0] + B_u[1] * power_curr + B_u[2] * fuel_curr > 0)
                #sw_dec_d = (B_d[0] + B_d[1] * power_curr + B_d[2] * fuel_curr > 0)

                # current values updated 
                #V_d = sw_dec_d * V_d_u + (1-sw_dec_d) * V_d_d 
                #V_u = sw_dec_u * V_u_d + (1-sw_dec_u) * V_u_u
                
            #V_d_all[:, month_nb] = V_d
            #V_u_all[:, month_nb] = V_u

        return (B_d, B_u, V_d_all, V_u_all)



# regression - x has to be a matrix 
def regress(y,x):
    nb_vars = x.shape[1]
    if nb_vars == 1:
        return sum(x * y)/sum (x * x)
    else:
        return solve(dot(x.transpose(), x)  , dot (x.transpose(), y))


# regression - x has to be a matrix, y,x are both gpu arrays
def regress_gpu(y,x):
    nb_vars = x.shape[1]
    if nb_vars == 1:
        return gpa.sum(x * y,dtype=float)/gpa.sum (x**2,dtype=float)
    else: # using 
        return solve(dot(x.transpose(), x)  , dot (x.transpose(), y))

