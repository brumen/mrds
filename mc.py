import config
import numpy as np
import scipy.integrate
from numpy.random import multivariate_normal as mn_cpu
import vtpm_cpu  # avx & omp analysis
if config.CUDA_PRESENT:
    import pycuda.cumath
    import cuda_ops as co
    import pycuda.gpuarray as gpa
    import pycuda.cumath as cumath
    import curand
    rn_gen_global = curand.create_gen_simple()  # generator of random numbers
else:
    rn_gen_global = None


def mc_one_step(F_v, s_v, T_v, rho_m, nb_sim, model='ln'):
    """
    one step simulation:
    F_v ... vec. of forwards
    s_v ... vec. of vols
    T_v ... vec. of simulation times
    rho_m ... correlation matrix
    nb_sim ... number of simulations.
    """
    if model is 'ln':
        if type(F_v) == float:
            rn = np.sqrt(T_v) * np.random.normal(size=nb_sim)
        else:
            rn = np.sqrt(T_v) * mn(np.zeros(len(F_v)), rho_m, size=nb_sim)
        F_sim = F_v * np.exp(s_v * rn - 0.5 * s_v **2 * T_v)
        return F_sim
    else:
        if np.isscalar(F_v):
            sZ = s_v * T_v * np.random.normal(0., 1., size=nb_sim)
        else:
            sZ = s_v * T_v * np.random.normal(0., 1., size=(nb_sim, 1))  # 1 factor model
        return F_v + sZ


def integrate_fct(s_fct, T_start, T_end, ttm, params,
                  drift_vol_ind='vol',
                  model='ln'):
    # integrates the drift between the T_start and T_end where T_start is ttm away from maturity
    # d_fct is the integrating function
    if model == 'ln':  # TO CORRECT TO CORRECT ---- ADD PARAMETERS TO S
        if drift_vol_ind == 'vol':
            integr_val = scipy.integrate.quad(lambda x: s_fct(ttm-(x-T_start))**2, T_start, T_end)[0] / (T_end - T_start)
            integr_val = np.sqrt(integr_val)
        else:  # drift integration
            integr_val = scipy.integrate.quad(lambda x: s_fct(ttm-(x-T_start)), T_start, T_end)[0] / (T_end - T_start)
    return integr_val


def mc_mult_steps_cpu(F_v, s_v, T_l, rho_m, nb_sim,
                      T_v_exp,
                      ao_f=None, ao_p=None,
                      d_v=None, 
                      cva_vals=None, model='n',
                      F_ret=None,
                      cuda_ind=False,
                      gen_first=True):
    """
    same as above,
    :param F_v: list of forward values
    :param s_v: vector of vols for forward vals
    :param T_l: list of time points at which all F_v should be simulated
    :param rho_m: correlation matrix
    :param nb_sim: number of sims
    :param T_v_exp: expiry of the forward contracts 
    :param ao_f: function to compute new vals from old ones
    :param ao_p: initial parameters
    :param d_v: drift of the process 
    :param cva_vals: None ... everything is regenerated anew
                     a list of random numbers generated which can be reused all the time
    :param model: model 'ln' or 'n' for normal
    :param F_ret: a sumulated list of return values - size (nb_sim, 1)
    :param cuda_ind: cuda inidcator 
    returns:
       matrix [time_step, simulation, fwd] or new parameters
    """
    if not cuda_ind:
        mn = mn_cpu
    else:
        mn = mn_gpu

    if T_l[0] != 0.:
        T_l_local = np.zeros(len(T_l)+1)
        T_l_local[1:] = T_l
    else:
        T_l_local = T_l

    nb_fwds = len(F_v)
    T_l_diff = np.diff(T_l_local)

    if not cuda_ind:
        F_sim_prev = np.empty((nb_fwds, nb_sim))
    else:
        F_sim_prev = gpa.empty((nb_fwds, nb_sim), np.double)

    F_sim_next = np.empty((nb_fwds, nb_sim))
        
    if ao_f is None:
        if not cuda_ind:
            F_sim_l = np.empty((len(T_l_local), nb_fwds, nb_sim))
        else:
            F_sim_l = gpa.empty((len(T_l_local), nb_fwds, nb_sim), np.double)
    else:
        F_sim_l = None
    F_v_shape = F_v.shape
    # assumption F_v and s_v are in the same format
    if len(F_v_shape) == 1 or F_v_shape[0] == 1:
        F_v_used = F_v.reshape((nb_fwds, 1))  # F_v_used is column vector 
    else:
        F_v_used = F_v

    if not cuda_ind:
        F_sim_prev[:, :] = F_v_used
    else:  # write in F_sim_prev F_v_used by columns
        F_sim_prev = co.set_mat_by_vec(F_v_used, nb_sim)
        
    if ao_f is None:
        F_sim_l[0, :, :] = F_sim_prev 
    else:
        ao_p_next = ao_p
        
    for T_ind, (T_diff, T_curr) in enumerate(zip(T_l_diff, T_l_local[:-1])):
        ttm_used = np.array(T_v_exp) - T_curr
        s_v_used = np.array([integrate_fct(s_vol, T_curr, T_curr+T_diff, ttm_curr, None, drift_vol_ind='vol')
                             for (s_vol, ttm_curr) in zip(s_v, ttm_used)])  # volatility depends on time-to-maturity

        if d_v is not None:
            d_v_used = np.array([integrate_fct(d_v_curr, T_curr, T_curr+T_diff, ttm_curr, None, drift_vol_ind='drift')
                                 for (d_v_curr, ttm_curr) in zip(d_v, ttm_used)])
        if not cuda_ind:
            s_v_used = s_v_used.reshape((len(s_v_used), 1))
            d_v_used = d_v_used.reshape((len(d_v_used), 1))
        
        if nb_fwds == 1:
            rn_sim_l = np.random.normal(size=(nb_sim, 1))
        else:
            if cva_vals is None:
                if not cuda_ind:
                    rvs = mn(np.zeros(nb_fwds), rho_m, size=nb_sim).transpose()
                else:
                    rvs = mn_gpu(0, rho_m, size=nb_sim)
            else:
                rvs = cva_vals[T_ind]
            rn_sim_l = rvs

        if model == 'ln':  # log-normal model 
            if not cuda_ind:
                expon_part = (np.sqrt(T_diff) * s_v_used) * rn_sim_l - 0.5 * s_v_used**2 * T_diff
                if d_v is not None:
                    expon_part += d_v_used * T_diff
                F_sim_next = F_sim_prev * np.exp(expon_part)
            else:  # gpu
                co.vtpm_cols_new_hd_ao(- 0.5 * s_v_used**2 * T_diff, np.sqrt(T_diff) * s_v_used, rn_sim_l)
                F_sim_part = rn_sim_l
                if d_v is not None:
                    F_sim_part = co.vtpm_cols_new_hd(d_v_used * T_diff, F_sim_part, tm_ind='p')
                # F_sim_next = F_sim_prev * pycuda.cumath.exp(F_sim_part)
                co.sin_cos_exp_d(F_sim_part, F_sim_part, sin_cos_exp='exp')  # works faster than cumath
                F_sim_next = F_sim_prev * F_sim_part
        else:  # normal model
            if not cuda_ind:
                # F_sim_next = F_sim_prev + s_v_used * np.sqrt(T_diff) * rn_sim_l
                # if d_v is not None:
                #    F_sim_next += d_v_used * T_diff
                vtpm_cpu.vm_ao(F_sim_prev, d_v_used * T_diff,
                               s_v_used * np.sqrt(T_diff), rn_sim_l, F_sim_next,
                               F_sim_prev.shape[0], F_sim_prev.shape[1])
            else:
                # addition 
                F_sim_next = co.vtpm_cols_new_hd(s_v_used * np.sqrt(T_diff), rn_sim_l, tm_ind='t')
                if d_v is not None:
                    F_sim_next = co.vtpm_cols_new_hd(d_v_used * T_diff, F_sim_next, tm_ind='p')
                F_sim_next += F_sim_prev  # + rn_sim_l

        if ao_f is None:
            F_sim_l[T_ind+1, :, :] = F_sim_next
            if F_ret is not None:
                F_sim_l[T_ind+1, :, :] += F_ret
        else:  # relevant case 
            if F_ret is None:
                ao_p_next = ao_f(F_sim_next, ao_p_next, cuda_ind=cuda_ind)
            else:
                if not cuda_ind:
                    F_sim_next_used = F_sim_next + F_ret
                else:
                    # F_dep_plus_ret = F_sim_next + F_ret
                    # F_dep_plus_ret = co.vtpm_cols_new(F_ret, F_sim_next)
                    # F_dep_plus_ret = co.vtpm_rows_new_ao(F_ret, F_sim_next)
                    co.vtpm_rows_new_ao(F_ret, F_sim_next)  # WRONG WRONG WRONG WRONG WRONG WRONG 
                    F_sim_next_used = F_sim_next  # WRONG WRONG WRONG 
                # ao_p_next = ao_f(F_dep_plus_ret, ao_p_next, cuda_ind=cuda_ind)
                ao_p_next = ao_f(F_sim_next_used, ao_p_next, cuda_ind=cuda_ind)
        F_sim_prev = F_sim_next

    if ao_f is None:
        if T_l[0] != 0.:
            return F_sim_l[1:, :, :]
        else:
            return F_sim_l
    else:
        return ao_p_next 


def ao_compute_from_F(F_sim_l, T_l,
                      ao_f, ao_p,
                      cuda_ind=False):
    """
    computes the ao_f vectors from previously simulated F_sim_l
    :param F_sim_l: shape of  (len(T_l), nb_fwds, nb_sim))
    :param T_l: list of time points at which all F_v should be simulated
    :param ao_f: function to compute new vals from old ones
    :param ao_p: initial parameters
    :param cuda_ind: cuda inidcator 
    returns:
       ao_p_next
    """
    F_sim_prev = F_sim_l[0, :, :]
    ao_p_next = ao_p
        
    for T_ind in range(len(T_l) - 1): 
        F_sim_next = F_sim_l[T_ind+1, :, :]
        ao_p_next = ao_f(F_sim_next, ao_p_next, cuda_ind=cuda_ind)
        F_sim_prev = F_sim_next

    return ao_p_next 

    

def mc_mult_steps_cpu_ret(F_v, s_v, T_l, rho_m, nb_sim,
                          T_v_exp, 
                          ao_f=None, ao_p=None,
                          d_v=None, 
                          cva_vals=None, model='n',
                          cuda_ind=False,
                          gen_first=False):
    """
    same as above,
    :param F_v: tuple of list of forward values, first for departure, second for return 
    :param s_v: tuple of IMPROVE HERE vector of vols for forward vals
    :param T_l: list of time points at which all F_v should be simulated
    :param rho_m: correlation matrix
    :param nb_sim: number of sims
    :param T_v_exp: expiry of forward contracts
    :param ao_f: function to compute new vals from old ones
    :param ao_p: initial parameters
    :param d_v: drift of the process, drift is a function of 
       d_v[i] is a function of (ttm, params)
    :param cva_vals: None ... everything is regenerated anew
                     a list of random numbers generated which can be reused all the time
    :param model: model 'ln' or 'n' for normal
    :param cuda_ind: whether cuda or cpu is used 
    :param gen_first: generates the return simulations first, then only runs over them 
    returns:
       matrix [time_step, simulation, fwd] or new parameters
    """
    if not cuda_ind:
        mn = mn_cpu
    else:
        mn = mn_gpu

    def add_zero_to_Tl(T_list):
        if T_list[0] != 0.:
            T_l_local = np.zeros(len(T_list)+1)
            T_l_local[1:] = T_list
        else:
            T_l_local = T_list
        return T_l_local

    F_v_dep, F_v_ret = F_v
    s_v_dep, s_v_ret = s_v  # a function
    if d_v is not None:
        d_v_dep, d_v_ret = d_v  # a function 
    T_l_dep, T_l_ret = add_zero_to_Tl(T_l[0]), add_zero_to_Tl(T_l[1])
    T_v_exp_dep, T_v_exp_ret = T_v_exp  # expiry values 
    rho_m_dep, rho_m_ret = rho_m
    
    nb_fwds_dep, nb_fwd_ret = len(F_v_dep), len(F_v_ret)
    T_l_diff_dep = np.diff(T_l_dep)
    if not cuda_ind:
        F_sim_prev = np.empty((nb_fwds_dep, nb_sim))
        max_fct_used = np.maximum
    else:
        F_sim_prev = gpa.empty((nb_fwds_dep, nb_sim), np.double)
        max_fct_used = gpa.maximum

    if ao_f is None:
        if not cuda_ind:
            F_sim_l = np.empty((len(T_l_dep), nb_fwds_dep, nb_sim))
        else:
            F_sim_l = gpa.empty((len(T_l_dep), nb_fwds_dep, nb_sim))
    else:
        F_sim_l = None
    F_v_shape = F_v_dep.shape
    # assumption F_v and s_v are in the same format
    if len(F_v_shape) == 1 or F_v_shape[0] == 1:
        F_v_used = F_v_dep.reshape((nb_fwds_dep, 1))
        T_v_used = T_v_exp_dep.reshape((nb_fwds_dep, 1))
    else:
        F_v_used = F_v_dep
        T_v_used = T_v_exp_dep

    if not cuda_ind:
        F_sim_prev[:, :] = F_v_used
    else:
        F_sim_prev = co.set_mat_by_vec(F_v_used, nb_sim)

    F_sim_next = np.empty((nb_fwds_dep, nb_sim))
        
    if ao_f is None:
        F_sim_l[0, :, :] = F_sim_prev 
    else:
        ao_p_next = ao_p

    if d_v is not None:
        d_v_ret_used = d_v_ret
    else:
        d_v_ret_used = None

    if gen_first:  # generates the return simulations & stores them
        F_sim_ret = mc_mult_steps_cpu(F_v_ret, s_v_ret, T_l_ret,
                                      rho_m_ret, nb_sim,
                                      T_v_exp_ret,
                                      # T_v_exp_ret_ch, 
                                      d_v=d_v_ret_used, 
                                      cva_vals=cva_vals, model=model,
                                      cuda_ind=cuda_ind)
        
    for T_ind, (T_diff, T_curr) in enumerate(zip(T_l_diff_dep, T_l_dep[:-1])):  # current time 
        ttm_dep = np.array(T_v_used) - T_curr  # time to maturity for departures 
        s_v_used = np.array([integrate_fct(s_vol, T_curr, T_curr+T_diff, ttm_curr, None, drift_vol_ind='vol')
                             for (s_vol, ttm_curr) in zip(s_v_dep, ttm_dep)])  # volatility depends on time-to-maturity
        if d_v is not None:
            d_v_used = np.array([integrate_fct(d_v_curr, T_curr, T_curr+T_diff, ttm_curr, None, drift_vol_ind='drift') for (d_v_curr, ttm_curr) in
                                 zip(d_v_dep, ttm_dep)])
            
        if not cuda_ind:
            s_v_used = s_v_used.reshape((len(s_v_used), 1))
            d_v_used = d_v_used.reshape((len(d_v_used), 1))
        
        if nb_fwds_dep == 1:
            rn_sim_l = np.random.normal(size=(nb_sim_dep, 1))
        else:
            if cva_vals is None:
                if not cuda_ind:
                    rvs = mn(np.zeros(nb_fwds_dep), rho_m_dep, size=nb_sim).transpose()
                else:
                    rvs = mn_gpu(0, rho_m_dep, size=nb_sim)
            else:
                rvs = cva_vals[T_ind]
            rn_sim_l =  rvs

        if model == 'ln':  # log-normal model 
            if not cuda_ind:
                # F_sim_next = F_sim_prev * np.exp(s_v_used * rn_sim_l - 0.5 * s_v_used**2 * T_diff)
                expon_part = (np.sqrt(T_diff) * s_v_used) * rn_sim_l - 0.5 * s_v_used**2 * T_diff
                if d_v is not None:
                    expon_part += d_v_used * T_diff
                F_sim_next = F_sim_prev * np.exp(expon_part)
            else:
                co.vtpm_cols_new_hd_ao(- 0.5 * s_v_used**2 * T_diff, np.sqrt(T_diff) * s_v_used, rn_sim_l)
                F_sim_part = rn_sim_l
                if d_v is not None:
                    F_sim_part = co.vtpm_cols_new_hd(d_v_used * T_diff, F_sim_part, tm_ind='p')
                # F_sim_next = F_sim_prev * pycuda.cumath.exp(F_sim_part)
                co.sin_cos_exp_d(F_sim_part, F_sim_part, sin_cos_exp='exp')
                F_sim_next = F_sim_prev * F_sim_part
        else:  # normal model (not updated completely)
            if not cuda_ind:
                # these three lines below fully work
                # F_sim_next = F_sim_prev + s_v_used * np.sqrt(T_diff) * rn_sim_l
                # if d_v is not None:
                #     F_sim_next += d_v_used * T_diff
                vtpm_cpu.vm_ao(F_sim_prev, d_v_used * T_diff,
                               s_v_used * np.sqrt(T_diff), rn_sim_l, F_sim_next,
                               F_sim_prev.shape[0], F_sim_prev.shape[1])
            else:
                F_sim_next = co.vtpm_cols_new_hd(s_v_used * np.sqrt(T_diff), rn_sim_l, tm_ind='t')
                if d_v is not None:
                    F_sim_next = co.vtpm_cols_new_hd(d_v_used * T_diff, F_sim_next, tm_ind='p')
                F_sim_next += F_sim_prev
                
        if ao_f is None:
            F_sim_l[T_ind+1, :, :] = F_sim_next
        else:
            # T_v_exp_ret_ch = T_v_exp_ret - T_curr
            for F_dep_idx in range(len(F_v_dep)):
                if not gen_first: 
                    ao_p_tmp = mc_mult_steps_cpu(F_v_ret, s_v_ret, T_l_ret,
                                                 rho_m_ret, nb_sim,
                                                 # T_v_exp_ret_ch,
                                                 T_v_exp_ret, 
                                                 ao_f=ao_f, ao_p=ao_p_next,
                                                 d_v=d_v_ret_used, 
                                                 cva_vals=cva_vals, model=model,
                                                 # F_ret is simulations of that departure flight
                                                 F_ret=F_sim_next[F_dep_idx, :],
                                                 cuda_ind=cuda_ind)
                else:
                    ao_p_tmp = ao_compute_from_F(F_sim_ret + F_sim_next[F_dep_idx, :].reshape((1, 1, nb_sim)),  # latter is vector (row)
                                                 T_l_ret, ao_f, ao_p,
                                                 cuda_ind=False)

                if not cuda_ind:
                    # ao_p_next['F_max_prev'] = max_fct_used(ao_p_tmp['F_max_prev'],
                    #                                       ao_p_next['F_max_prev'])
                    aop_rows, aop_cols = ao_p_next['F_max_prev'].shape
                    vtpm_cpu.max2m(ao_p_tmp['F_max_prev'],
                                   ao_p_next['F_max_prev'],
                                   ao_p_next['F_max_prev'],
                                   aop_rows, aop_cols)
                else:
                    ao_p_next['F_max_prev'] = max_fct_used(ao_p_tmp['F_max_prev'],
                                                           ao_p_next['F_max_prev'])
                    
        F_sim_prev = F_sim_next

    if ao_f is None:
        if T_l[0] != 0.:
            return F_sim_l[1:, :, :]
        else:
            return F_sim_l
    else:
        return ao_p_next 


def mc_mult_steps(F, s, T_l, rho_m, nb_sim, cva_vals=None, ci=False):
    """
    :param F: for CPU - np.array of forwards, for GPU - gpuarray of forwards
    :param s: similar as above
    :param ci: cuda indicator
    """
    if not ci:
        return mc_mult_steps_cpu(F, s, T_l, rho_m, nb_sim, cva_vals=cva_vals)
    else:  # cuda
        return mc_mult_steps_cuda(F, s, T_l, rho_m, nb_sim, rng=rn_gen_global,
                                  cva_vals=cva_vals)


def mn_gpu(unimp, rho_m, size=100):
    """
    :param unimp: unimportant, so that it coincides w/ mn from numpy
    :param rho_m: correlation matrix 
    :param size: number of simulation variables 
    standard multivariate normal with rho_m as correlation variable  
    """
    nb_fwds = rho_m.shape[0]
    sim_rn_init = gpa.empty((nb_fwds, size), dtype=np.double)
    rho_m_chol_cuda = gpa.to_gpu(np.linalg.cholesky(rho_m))
    simulated_rn = gpa.empty((nb_fwds, size), dtype=np.double)
    curand.gen_eff_dev_rns_double(sim_rn_init.size,
                                  np.longlong(sim_rn_init.ptr), rn_gen_global)
    co.matmul(rho_m_chol_cuda, sim_rn_init, simulated_rn)

    return simulated_rn
