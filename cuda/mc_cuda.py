def mc_one_step_cuda(F, s, T, nb_sim, rng=rn_gen_global):
    """
    one step simulation, just done on cuda 
    :param F: forward value
    :param s: volatility
    :param T: simulation time
    :param rng: random number generator, default used
    """
    rn_sim = gpa.empty(nb_sim, dtype=np.float32)
    curand.gen_eff_dev_rns(nb_sim, np.longlong(rn_sim.ptr), rng)
    Fsim = pycuda.cumath.exp(rn_sim * s * np.sqrt(T) - 0.5 * s**2 * T) * F
    return Fsim


def mc_mult_steps_cuda(F_v, s_v, T_l, rho_m, nb_sim,
                       d_v=None,
                       rng=rn_gen_global,
                       cva_vals=None,
                       model='ln'):
    """ 
    generate multiple steps of monte carlo on cuda
    :param F: device vector of forward values
    :param T_l: list of time points at which all F_v should be simulated, on host
    :param s: volatility of the forwards, on device
    :param rho_m: correlation matrix, on host
    returns:
       list [simulation, fwd] x time_step
    """
    if T_l[0] != 0.:
        T_l_local = np.zeros(len(T_l)+1)
        T_l_local[1:] = T_l
    else:
        T_l_local = T_l

    dtype = F_v.dtype
    if dtype == np.float32:
        gen_fct = curand.gen_eff_dev_rns
    else:
        gen_fct = curand.gen_eff_dev_rns_double
    nb_fwds = len(F_v)
    T_l_diff = np.diff(T_l_local)
    T_l_local_len = len(T_l_local)
    ones_d = cuda_ops.gpu_set_constant(nb_sim, 1.)
    F_sim_l = [[]] * T_l_local_len
    rn_sim_l_cum = [[]] * T_l_local_len
    # F_sim_l[0, :, :] = F_v
    F_sim_l[0] = cuda_ops.vtpv(F_v, ones_d, tm_ind='t')

    if cva_vals is None:
        rho_chol = gpa.to_gpu(np.linalg.cholesky(rho_m))
        rn_sim_l_cum_init = gpa.empty((nb_fwds, nb_sim))
        gen_fct(nb_sim * nb_fwds, np.longlong(rn_sim_l_cum_init[0].ptr), rng)
        rn_sim_l_cum[0] = gpa.empty((nb_fwds, nb_sim))
        cuda_ops.matmul(rho_chol, rn_sim_l_cum_init, rn_sim_l_cum[0])
        new_rn_init = gpa.empty((nb_fwds, nb_sim))
        new_rn = gpa.empty((nb_fwds, nb_sim))
    else:
        rn_sim_l_cum[0] = cva_vals[0]

    for T_ind, T_diff in enumerate(T_l_diff):
        if cva_vals is None:
            gen_fct(nb_sim * nb_fwds, np.longlong(new_rn_init.ptr), rng)
            cuda_ops.matmul(rho_chol, new_rn_init, new_rn)
        else:
            new_rn = cva_vals[T_ind+1]

        rn_sim_l_cum[T_ind+1] = rn_sim_l_cum[T_ind] + new_rn * np.sqrt(T_diff)
        s_v_mult = cuda_ops.vtpm_cols_new(s_v, rn_sim_l_cum[T_ind+1], tm_ind='t')
        if model == 'ln':  # log-normal model
            s2T = s**2 * T_l_local[T_ind+1] * (-0.5)
            s_v_mult_ded = cuda_ops.vtpm_cols_new(s2T, s_v_mult, tm_ind='p')
            s_v_exp = cumath.exp(s_v_mult_ded)
            F_sim_l[T_ind+1] = cuda_ops.vtpm_cols_new(F_v, s_v_exp, tm_ind='t')
        else:
            F_sim_l[T_ind+1] = cuda_ops.vtpm_cols_new(F_v, s_v_exp, tm_ind='t')
            if d_v != None:
                F_sim_l[T_ind+1] += d_v[T_ind] * T_diff
            
    if T_l[0] != 0.:
        return F_sim_l[1:]
    else:
        return F_sim_l
