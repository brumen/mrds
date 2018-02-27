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
from extendible import extend_opt as eo

nb_fwds = 48
power_f = np.linspace (40, 30, nb_fwds)
gas_f = np.ones(nb_fwds) * 5.
power_v = np.linspace (0.4, 0.3, nb_fwds)
gas_v = np.linspace (0.4, 0.3, nb_fwds)
power_gas_v = np.array([power_v, gas_v]).reshape(2*nb_fwds)
power_gas_f = np.array([power_f, gas_f]).reshape(2*nb_fwds)
HR = 7
corr_m = vols.corr_hyp_sec_mat (0.99, np.arange(2*nb_fwds))
T_l = np.array([0.1, 0.2, 0.5])
opt_exp = 1.
premium = 0.

nb_sim = 10
nb_sim_pricer = 200000
nb_sim_paths = 50


# testing mc_one_step
# rn = np.random.normal(size =(3, len(power_f)))
# print "cpu = ", mc.mc_one_step_nocuda(power_f, power_v, 1., rn)
# rn_d = gpa.to_gpu (rn.astype(np.float32))
# print "gpu = ", mc.mc_one_step_cuda(gpa.to_gpu (power_f.astype(np.float32)),
#                                     gpa.to_gpu (power_v.astype(np.float32)),
#                                     1., rn_d)



# a2 = mc.mc_mult_steps (power_gas_f, power_gas_v, T_l, corr_m, 100)

# t1 = time.time()
# exp_gpu = eo(power_gas_f, power_gas_v, HR, premium, corr_m, T_l, opt_exp, 
#              nb_sim_paths, nb_sim_pricer, 
#              cuda_ind = True)
# print "GPU time = ", time.time() - t1
# print "GPU exp = ", np.mean(exp_gpu, axis = 0)

# t1 = time.time()
# exp_cpu = extend_opt(power_gas_f, power_gas_v, HR, premium, corr_m, T_l, opt_exp, 
#                          nb_sim_paths, nb_sim_pricer, 
#                          cuda_ind = False)
# print "CPU time = ", time.time() - t1
# print "CPU exp = ", np.mean(exp_cpu,axis=0)

# performing the grid calibration 

T_l = np.array([0.])

val_K = [ eo(power_gas_f, power_gas_v, HR, premium, corr_m, T_l, opt_exp, 
             1, nb_sim_pricer, 
             cuda_ind = False)  
          for premium in np.linspace (0, 400, 15)]

# Inverse volatility estimation. 


# Compare the actual exposure at T_l w/ the Black's value for those times. 
T_l = np.array([0.1, 0.2, 0.5])

