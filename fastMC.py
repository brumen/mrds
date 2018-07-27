# implements the Ninomiya MC engine 

from numpy import *
import numpy.random
from scipy.integrate import ode
import multiprocessing


def ninomiya_victoir_step ( V0, V1, Xk, k, dt, params ):  # Y(t_k) = Xk
    """
    implements the ninomiya-ninomiya step

    V0 ... V0 (x,t) drift function
    V1 ... V1 (x,t) diffusion function
    Xk ... inital state Y(t_k)
    k ... time step
    dt ... time step
    params ... [p0, p1] ... parameters for V0, V1
    """

    V0new = lambda t,y, params : V0 (t,y,params) * dt / 2
    # binomial_sample = 2 * rbinom (1, 1, 0.5) -1  ## bernoulli rand. variable +/- 1 w.p. 1/2
    V1new = lambda t, y, params: V1 (t,y,params) * sqrt (dt) * numpy.random.normal() # * binomial_sample 

    # simple implementation of runge-kutta 
    rk_int_1 = ode(V0new).set_integrator('zvode', method='bdf', with_jacobian=False)
    rk_int_1.set_initial_value(Xk,0.).set_f_params(params[0])
    res1 = rk_int_1.integrate(rk_int_1.t + dt )

    rk_int2 = ode(V1new).set_integrator('zvode', method='bdf', with_jacobian=False)
    rk_int2.set_initial_value(res1,0.).set_f_params(params[1])
    res2 = rk_int2.integrate(rk_int2.t + dt) 
      
    rk_int3 = ode(V0new).set_integrator('zvode', method='bdf', with_jacobian=False)
    rk_int3.set_initial_value(res2, 0.).set_f_params(params[0])
    res3 = rk_int3.integrate(rk_int3.t + dt)
  
    return res3


#
# computes the X_{k+1} from X_k using the Ninomiya-Victoir method
#   V0, V1 are functions, such that the sde is given by
#   dY = V_0 (t,Y) dt + V_1 (t, Y) dW
#   innov should be replaced by numpy.random.standard_normal()
def wong_zakai_step ( V_0, V_1, X_k, t_k, dt, innov, params ):  # Y(t_k) = Xk 
    
    Vstep = lambda t,y,params: V_0 (t,y,params) * dt + \
            V_1 (t,y,params) * sqrt (dt) * innov

    # simple implementation of runge-kutta 
    r = ode(Vstep).set_integrator('zvode', method='bdf', with_jacobian=False)
    r.set_initial_value (X_k,t_k).set_f_params(params)

    r.integrate(r.t+dt)
    return r.y.real[0]
    

# brute-force on runge-kutta 
# innov should be replaced by numpy.random.standard_normal ()
def wong_zakai_step_brute ( V_0, V_1, X_k, t_k, dt, innov, params ):  # Y(t_k) = X_k 
    
    Vstep = lambda t,y,params: V_0 (t,y,params) * dt + \
            V_1 (t,y,params) * sqrt (dt) * innov

    # brute-force runge-kutta
    k_1 = dt * Vstep (t_k, X_k, params)
    k_2 = dt * Vstep (t_k + dt/2.0, X_k + k_1/2.0, params)
    k_3 = dt * Vstep (t_k + dt/2.0, X_k + k_2/2.0, params)
    k_4 = dt * Vstep (t_k + dt, X_k + k_3, params)
    
    return X_k + k_1/6.0 + k_2/3.0 + k_3/3.0 + k_4/6.0


# performs the nb_steps of the wong_zakai method 
def wong_zakai_path (V_0, V_1, X_0, t_0, dt, nb_steps, params):

    t_line = t_0 + dt * numpy.arange (nb_steps)
    red_fct = lambda X, i_Z: wong_zakai_step (V_0, V_1, X, t_line[i_Z[0]], \
                                              dt, i_Z[1], params)
    Z_vec = numpy.random.standard_normal(nb_steps)
    i_Z_vec = zip (t_line, Z_vec)

    return reduce (red_fct, i_Z_vec, X_0) # left fold CHECK IF THIS IS CORRECT

def wong_zakai_distr (V_0, V_1, X_0, t_0, dt, nb_steps, nb_repeats, params):
    return map ( lambda x: wong_zakai_path (V_0, V_1, X_0, t_0, dt, nb_steps, params), range (nb_repeats) )


def wong_zakai_distr_multiproc (V_0, V_1, X_0, t_0, dt, nb_steps, nb_repeats, params):
    nb_cores = mul.cpu_count()
    print "Nb. cores used is ", nb_cores
    pool = mul.Pool(processes=nb_cores)

    def wong_zakai_path_simple(x):
        return wong_zakai_path (V_0, V_1, X_0, t_0, dt, nb_steps, params)

    return pool.map ( wong_zakai_path_simple, range (nb_repeats) )





#
# generates one simulation path from MConestep 
def MConepath (V0, V1, X0, T, nb_time_steps, params):
    dt = T / double (nb_time_steps)
    MConestep_tmp = lambda x, k, params: ninomiya_victoir_step (V0, V1, x, k, dt, params)
    one_path = zeros(nb_time_steps)
    one_path[0] = X0
    for ts in xrange (1,nb_time_steps):
        one_path[ts] = one_path[ts-1] + ninomiya_victoir_step (V0, V1, one_path[ts-1], ts-1, dt, params)[0]

    return one_path

# same as above, but multithreaded 
def MConepath_thread (V0, V1, X0, T, nb_time_steps, params, nb_sims):
    nb_cores = multiprocessing.cpu_count()
    print "Using ", nb_cores , " cores."
    pool = multiprocessing.Pool(processes=nb_cores) 
    print "Starting worker threads"
    C = pool.map ( MConepath, zip ([V0] * nb_sims, [V1] * nb_sims, [X0] * nb_sims, [T] * nb_sims, [nb_time_steps] * nb_sims,
        [params] * nb_sims ) )
    return C





#
#  path_stat_fun is the path statistic for each independent path
#   sw ... 1 for behavioral model , 2 for optimal model
def AverageOverPaths (value_fct, V0, V1, X0, T, nb_time_steps, nb_simulations, sw, params):
    
    tmp_res = 0
    for ind1 in xrange(1,nb_simulations):
        opath_tmp = MConepath1 (V0, V1, X0, T, nb_time_steps, params) 
        loan_val_tmp <- loan_val_fct (1, params, sw, path_tmp )
        tmp_res = tmp_res + loan_val_tmp 

    return tmp_res / nb_simulations


# this one returns the whole list of functions 
def AverageOverPaths2 (loan_val_fct, V0, V1, X0, T, \
                       nb_time_steps, nb_simulations, sw, params):

    tmp_res = array([0])
    for ind1 in xrange(nb_simulations):
        path_tmp = MConepath1 (V0, V1, X0, T, nb_time_steps, params) 
        loan_val_tmp = loan_val_fct (1, params, sw, path_tmp )
        tmp_res.append ( loan_val_tmp )

    return tmp_res 

## 
## Euler scheme implementation for the loan pricer
def MConestep1 ( V0, V1, Xk, k, time_step, params ): ##Y(t_k) = Xk 
    
    jump_nb = rpois (1, params.lam * time_step )
    jump_sizes = rexp (jump_nb, rate = 1/ ( hazard_function (params.t_dates[k], params.alphas, params.M_vec) * params.nu ) )
  
    return Xk + V0(params.t_dates[k], Xk, params) * time_step + V1 (params.t_dates[k], Xk, params) * sqrt (time_step) * rnorm(1) + sum (jump_sizes) 
  

def MConepath1 (V0, V1, X0, T, nb_time_steps, params):
    time_step = T/ nb_time_steps  
    MConestep_tmp = function (x, k, params)
    MConestep1 (V0, V1, x, k, time_step, params)

    # CHECK THIS BELOW 
    return reduce (MConestep_tmp, nb_time_steps , params, X0 ) 


#
## simulates JCIR process with time steps 
def MConestep2 ( V0, V1, Xk, k, time_step, params ): ##Y(t_k) = Xk 
    
    jump_nb = rpois (1, params.lam * time_step )
    jump_sizes = rexp (jump_nb, rate = 1. / ( hazard_function ( k * time_step , params.alphas, params.M_vec) * params.nu ) )
  
    return Xk + V0( k * time_step , Xk, params) * time_step + V1 ( k * time_step, Xk, params) * sqrt (time_step) * rnorm(1) + sum (jump_sizes) 

