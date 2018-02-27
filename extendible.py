#
# extendible option on cuda
import config
import numpy as np
import openopt # optimization solver

import pycuda.curandom
import pycuda.gpuarray as gpa
import pycuda.cumath
import pycuda.reduction
from pycuda.compiler import SourceModule

#from cudanormal import cudanormal
import datetime
import time

import cublas
from cuda_ops import *

import mc
import vols


def extend_opt(power_gas_f, power_gas_v, HR, premium, corr_m, T_l, opt_exp, 
               nb_sim_paths, nb_sim_pricer, 
               cuda_ind = False):
    """
    extendible option CVA pricer:
    
    cuda_ind ... indicator whether cuda is used 
    """
    
    nb_fwds = len(power_gas_f)/2

    # random numbers, correctly correlated
    rn = np.random.multivariate_normal ( np.zeros (2*nb_fwds), 
                                         corr_m, size = nb_sim_pricer )
    rn_power = rn [:,:nb_fwds]
    rn_gas = rn [:, nb_fwds:]

    if cuda_ind:
        rn_power_d = gpa.to_gpu (rn_power.astype(np.float32))
        rn_gas_d = gpa.to_gpu (rn_gas.astype(np.float32))
        power_s_d_tmp = power_gas_v[:nb_fwds]
        power_s_d = gpa.to_gpu (power_s_d_tmp.astype(np.float32)) # _device_ vols for power
        gas_s_d = gpa.to_gpu (power_gas_v[nb_fwds:].astype(np.float32)) # _device_ vols for gas 

    sim_paths = mc.mc_mult_steps (power_gas_f, power_gas_v, T_l, corr_m, nb_sim_paths)

    sim_paths_results = np.zeros ((nb_sim_paths, len(T_l))) # results stored in this procedure

    for t_ind, t in enumerate(T_l):
        for path_ind in np.arange(nb_sim_paths):
            F_v_path = sim_paths[t_ind,path_ind,:]
            power_v_path = F_v_path[ : nb_fwds ]
            gas_v_path = F_v_path[ nb_fwds: ]

            # pricing simulations 
            if cuda_ind: # GPU
                power_sims = mc.mc_one_step_cuda (gpa.to_gpu(power_v_path.astype(np.float32)), power_s_d, 
                                                  opt_exp - t, rn_power_d )
                fuel_sims = mc.mc_one_step_cuda (gpa.to_gpu (gas_v_path.astype(np.float32)), gas_s_d, 
                                                 opt_exp - t, rn_gas_d )
                power_profits = power_sims - fuel_sims * HR - np.float32(premium)

                maximum_cuda ( power_profits )
                res_tmp = gpa.sum(power_profits).get() / float(nb_sim_pricer)
                sim_paths_results[path_ind, t_ind] = res_tmp

            else: # CPU 
                power_sims = mc.mc_one_step_nocuda (power_v_path, power_gas_v[:nb_fwds], 
                                                    opt_exp - t, rn_power )
                fuel_sims = mc.mc_one_step_nocuda (gas_v_path, power_gas_v[nb_fwds:], 
                                                   opt_exp - t, rn_gas )
                power_profits = np.sum (np.maximum (power_sims - HR * fuel_sims - premium, 0.)) / float(nb_sim_pricer)
                sim_paths_results[path_ind,t_ind] = power_profits

    return sim_paths_results

