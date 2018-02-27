# test of the tolling algorithm 
import config

import numpy
from numpy import *
import scipy
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats 
import scipy.optimize 

import xlrd # reading excel files 
import tolling 
import profile
import tolling_fast

import time


# cuda_ind ... indicator if cuda is present 
# tolling_fast_ind ... indicator if the routine should use fast pricing fct.
def setup_tolling_params_CMG(cuda_ind, tolling_fast_ind, tolling_fast_mt_ind): 

    nb_time_steps = 50
    nb_modes = 6

    d_dispatch_init = numpy.random.random((nb_modes,nb_time_steps))
    params = tolling.tolling_params()
    params.b_block_lengths = numpy.arange(nb_time_steps).reshape((1,nb_time_steps)) # block lengths
    params.nb_modes = nb_modes
    params.nb_time_steps = len (params.b_block_lengths)
    params.P = numpy.arange(nb_time_steps).reshape ( ( 1, nb_time_steps ) )
    params.c_matrix = numpy.random.random( ( nb_modes, nb_time_steps ) ) 
    params.h_matrix = numpy.random.random( ( nb_modes, nb_time_steps ) )
    params.B_matrix = numpy.random.random( ( nb_modes, nb_time_steps ) )
    params.v_matrix = numpy.random.random( ( nb_modes, nb_time_steps ) )
    params.K_startup_costs = numpy.random.random ( (1,nb_time_steps) )
    params.F_matrix = numpy.random.random ( (nb_modes, nb_modes) )
    params.w_matrix = numpy.random.random ( (nb_modes, nb_time_steps) )
    params.SF = 0.2
    params.E_S = 0.3
    params.MRL = 0.1
    params.MDT = 5 # minimum downtime 
    params.MUT = 5 # minimum uptime
    params.lattice_size = 120
    params.cuda = cuda_ind # indicator if cuda present or not 
    params.tolling_fast = tolling_fast_ind # indicator for the fast tolling agg.
    params.tolling_fast_mt = tolling_fast_mt_ind 

    params.F_1_0 = 100.
    params.F_2_0 = 95.
    params.nb_steps = 20

     # class construction and tests
    tm = tolling.tolling_model_CMG(d_dispatch_init, params)
    aaa = tm.prev_dispatch(d_dispatch_init)
    aa1 = tm.minimum_run_level(d_dispatch_init)
    aa2 = tm.SC_fixed_startup_costs(d_dispatch_init)
    aa3 = tm.SF_startup_fuel_costs(d_dispatch_init)
    aa4 = tm.SE_startup_energy()
    aa5 = tm.AC_additional_costs(d_dispatch_init)
    aa0 = tm.optimizing_fct(d_dispatch_init)
    aa8 = tm.minimum_downtime_uptime (d_dispatch_init, 1)

    return [params, d_dispatch_init]



 # cuda_ind & tolling_fast_ind both True is EXCLUSIVE 
def test_tolling_MRD_1 (cuda_ind, tolling_fast_ind):

    params, d_dispatch_init = setup_tolling_params_CMG(cuda_ind, tolling_fast_ind)

    tMRD = tolling.tolling_model_MRD(d_dispatch_init, params) 
    tMRD.transition_mtx (exp ( numpy.random.random((100,1)) ), 
                         exp ( numpy.random.random((100,1)) ), 
                         array([-1.1, 0.2, 0.3]), 
                         array([-1.14, 0.4, 0.3]) )

    tMRD.transition_mtx_multi ()
    tMRD.transition_mtx_internal()



def test_tolling_MRD_2(): 

    tMRD = tolling.tolling_model_MRD(d_dispatch_init, params) 
    tMRD.transition_mtx_internal()
    tMRD.terminal_pp_value()
    tMRD.step(1,0)
    tMRD.all_one_steps()
    tMRD.all_one_steps_mt()
    tMRD.multiple_steps(5)



def run_1 (cuda_ind, nb_steps, params):
    params1 = params
    params1.cuda = cuda_ind
    tMRD = tolling.tolling_model_MRD(d_dispatch_init, params) 
    tMRD.terminal_pp_value()
    tMRD.multiple_steps(nb_steps)

    return tMRD


def test4(cuda_ind, tolling_fast_ind, tolling_fast_mt_ind):
    params, d_dispatch_init = setup_tolling_params_CMG(cuda_ind, tolling_fast_ind, tolling_fast_mt_ind)
    tMRD = tolling.tolling_model_MRD(d_dispatch_init, params) 
    # tMRD.transition_mtx_internal()
    tMRD.terminal_pp_value()
    # tMRD.step(1,0)
    # profile.run('tMRD.all_one_steps()')
    # tMRD.all_one_steps_mt()
    # profile.run('tMRD.multiple_steps(3)')
    # profile.run('tMRD.multiple_steps_mt(10)')
    tMRD.multiple_steps(125)
    print "WORK = ", tMRD.work_pp_curr[1]
    print "IDLE = ", tMRD.idle_pp_curr[1]
    # tMRD.multiple_steps_mt(2)


    #
     #ti = 100
     #P_m = numpy.random.random ( (ti,ti,ti,ti))
     #H_m = numpy.random.random ( (ti,ti))
     #G_m = numpy.random.random ( (ti,ti))
     #res_m = numpy.random.random ( (ti,ti))
     #tolling_fast.tensor_fast (P_m, H_m, G_m, res_m )

def test5(cuda_ind, tolling_fast_ind, tolling_fast_mt_ind):
    params, d_dispatch_init = setup_tolling_params_CMG(cuda_ind, tolling_fast_ind, tolling_fast_mt_ind)
    tMRD = tolling.tolling_model_MRD(params) 

    # tMRD.transition_mtx_internal()

    # tMRD.terminal_pp_value()

    # tMRD.step(1,0)
    #  profile.run('tMRD.all_one_steps()')
    # tMRD.all_one_steps_mt()
    # profile.run('tMRD.multiple_steps(3)')
    # profile.run('tMRD.multiple_steps_mt(10)')
    # tMRD.multiple_steps(125)
    # print "WORK = ", tMRD.work_pp_curr[1]
    # print "IDLE = ", tMRD.idle_pp_curr[1]
    # tMRD.multiple_steps_mt(2)

    print tMRD.tolling_value()



#
# tests whether tensor products are implemented correctly 
#
def test_tensor_products(lattice_size = 64, tensor_chosen="tensor_P_m"):

    tensor_string = open ("tensor.cu").read()
    tensor_prod_mod = config.SourceModule(tensor_string% {"lss": lattice_size})
    tensor_prod = tensor_prod_mod.get_function ("tensor_P_m") # extracting compute vol function
    tensor_prod_alt = tensor_prod_mod.get_function ("tensor_P_m_alt2")

    P_m = numpy.random.random( ( lattice_size, lattice_size, lattice_size, lattice_size ) )
    H_m = numpy.random.random( ( lattice_size, lattice_size ) ) 
    G_m = numpy.random.random( ( lattice_size, lattice_size ) ) 
    res2_m = numpy.random.random( ( lattice_size, lattice_size ) ) 
    res4_m = numpy.random.random( ( lattice_size, lattice_size ) ) 


    P_d = config.gpuarray.to_gpu ( P_m ).astype(float32)
    H_d = config.gpuarray.to_gpu ( H_m ).astype(float32)
    G_d = config.gpuarray.to_gpu ( G_m ).astype(float32)
    res1_d = config.gpuarray.to_gpu ( numpy.random.random( ( lattice_size, lattice_size ) ) ).astype(float32)
    res3_d = config.gpuarray.to_gpu ( numpy.zeros( ( lattice_size, lattice_size ) ) ).astype(float32) # HAS TO BE 0 HERE. 

    for ind in range (10):
        t1 = time.time()
        tensor_prod(P_d, H_d, G_d, res1_d, block=(lattice_size,1,1), grid=(lattice_size,1))
        t1 = time.time() - t1
        print "Time 1 = ", t1

    res1_m = res1_d.get()
    
    for ind in range (10):
        t2 = time.time()
        tensor_prod_alt(P_d, H_d, G_d, res3_d, block=(32,1,1), grid=(lattice_size,lattice_size))
        t2 = time.time() - t2
        print "Time 2 = ", t2
         
    res3_m = res3_d.get()

    #tolling_fast.tensor_fast (P_m, H_m, G_m, res2_m) 
    t3 = time.time()
    tolling_fast.tensor_fast_mat (P_m, H_m, G_m, res4_m) 
    t3 = time.time() - t3
    print "Time 3 = ", t3


    print res1_m
    #print res2_m
    print res4_m
    print res3_m


# tests the daily simulation routine 
def test_daily_sim(cuda_ind, tolling_fast_ind, tolling_fast_mt_ind):
    params, d_dispatch_init = setup_tolling_params_CMG(cuda_ind, tolling_fast_ind, tolling_fast_mt_ind)
    tMRD = tolling.tolling_model_MRD(params) 

    return tMRD.daily_sim ([1,2,3],[0.2,0.3,0.4],[1.2,1.3,1.4],[1.,2.,3.], 100)


# simulate power and fuel 
def test_power_fuel():
    params, d_dispatch_init = setup_tolling_params_CMG(False, False, False)
    tMRD = tolling.tolling_model_MRD(params) 
    nb_sims = 600

    daily_sims_power = tMRD.daily_sim([30,32,33],[0.2,0.3,0.4],[1.2,1.3,1.4],[1.,2.,3.], nb_sims)
    daily_sims_fuel = tMRD.daily_sim([3.1,4.2,4.5],[0.25,0.35,0.45],[1.,1.,1.],[1.,2.,3.], nb_sims)

    # regression coefficients 
    regress_B = tMRD.daily_lsm (daily_sims_power, daily_sims_fuel)
    
    return regress_B


def test_power_fuel_cuda():
    params, d_dispatch_init = setup_tolling_params_CMG(False, False, False)
    tMRD = tolling.tolling_model_MRD(params) 
    nb_sims = 30

    daily_sims_power = tMRD.daily_sim_cuda([30,32,33],[0.2,0.3,0.4],[1.2,1.3,1.4],[1.,2.,3.], nb_sims)
    #print daily_sims_power
    daily_sims_fuel = tMRD.daily_sim_cuda([3.1,4.2,4.5],[0.25,0.35,0.45],[1.,1.,1.],[1.,2.,3.], nb_sims)

    # regression coefficients 
    regress_B = tMRD.daily_lsm_cuda (daily_sims_power, daily_sims_fuel)
    
    return regress_B
    
    #return daily_sims_power
