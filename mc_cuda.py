import config
import numpy as np
import scipy.integrate
from numpy.random import multivariate_normal as mn

if config.CUDA_PRESENT:
    import pycuda.cumath
    import cuda_ops 
    import pycuda.gpuarray as gpa
    import pycuda.cumath as cumath
    import curand
    rn_gen_global = curand.create_gen_simple()  # generator of random numbers
else:
    rn_gen_global = None


def mc_mult_steps_gpu(F_v, s_v, T_l, rho_m, nb_sim,
                      T_v_exp,
                      ao_f=None, ao_p=None,
                      d_v=None, 
                      cva_vals=None, model='ln',
                      F_ret=None):
    """
    simulate forward curves on cuda
    
    generates a 3-dimensional array
      0-th dimension: asset
      1-st dimension: simulation times
      2-nd dimension: curve
      3-rd dimension: repeats of the curve
    """
    np.random.seed(set_seed)
    fwd_c_len = len(F_v)
    nb_time_steps = len(T_l)
    sc = gpa.zeros((nb_time_steps, fwd_c_len, nb_sim), dtype=np.double)  # sc - simulated curve 
    fwd_c = F_v

    cuda_ops.vtpm_cols(np.log(fwd_c), sc[0, :, :], tm_ind='p')
    X = gpa.zeros((fwd_c_len, nb_sim), dtype=np.double)
    X_prev = gpa.empty((fwd_c_len[asset_nb], nb_simulations), dtype=np.double)
    simulated_rn = gpa.empty((fwd_c_len, nb_sim), dtype=np.double)
    compl_corr_chol_gpu = gpa.to_gpu(np.linalg.cholesky(rho_m))
    g1 = curand.create_gen_simple()

    # looping over time steps
    #   simulates ln process, basis for skew as well
    #   t_i ... idx of sim_time
    #   fact_sum ... factors of the individual assets
    for t_i in range(nb_time_steps):
            simulated_rn_init = gpa.empty((nb_factors, nb_simulations), dtype=np.double)
            curand.gen_eff_dev_rns_double(simulated_rn_init.size, np.longlong(simulated_rn_init.ptr), g1)
            cuda_ops.matmul(compl_corr_chol_gpu, simulated_rn_init, simulated_rn)

            for asset_nb in range(self.nb_assets):
                nb_factors_asset = self.nb_factors_list[asset_nb]
                old_cov_mat = self.complete_corr_mat[fact_sum[asset_nb]:fact_sum[asset_nb+1],
                                                     fact_sum[asset_nb]:fact_sum[asset_nb+1]]
                old_chol = np.linalg.cholesky(old_cov_mat)
                old_chol_inv = np.linalg.inv(old_chol)
                old_chol_inv_gpu = gpa.to_gpu(old_chol_inv)

                # extra matrices for using later
                sims_Z_unit = gpa.empty((fact_sum[asset_nb+1] - fact_sum[asset_nb], nb_simulations),
                                        dtype=np.double)
                sims_Z_unit_mul = gpa.empty((fact_sum[asset_nb+1] - fact_sum[asset_nb], nb_simulations),
                                            dtype=np.double)

                if tenor_list is None:
                    tenor_used = range(self.forward_curve_len[asset_nb])
                else:
                    tenor_used = tenor_list[asset_nb]

                for tenor_idx, tenor_nb in enumerate(tenor_used):
                    # prepare cov mtx
                    new_cov_mat = np.array([[self._var_covar_mtx(asset_nb, tenor_nb, i, j, t_i, self.simulation_times)
                                             for j in range(nb_factors_asset)]
                                            for i in range(nb_factors_asset)])
                    new_chol = np.linalg.cholesky(new_cov_mat)
                    new_chol_gpu = gpa.to_gpu(new_chol)

                    sims_Z = simulated_rn[fact_sum[asset_nb]:fact_sum[asset_nb+1], :]
                    cuda_ops.matmul(old_chol_inv_gpu, sims_Z, sims_Z_unit)
                    cuda_ops.matmul(new_chol_gpu, sims_Z_unit, sims_Z_unit_mul)
                    delta_X = cuda_ops.colsum_cuda_last(sims_Z_unit_mul)

                    # quadratic variation of delta_X
                    if t_i == 0:
                        t_prev_qv = 0.
                    else:
                        t_prev_qv = self.simulation_times[t_i - 1]
                    t_next_qv = self.simulation_times[t_i]
                    qv = np.sum([[self._V_cross_factor(asset_nb, factor_1, factor_2,
                                                       tenor_nb, tenor_nb, t_prev_qv, t_next_qv)
                                 for factor_1 in range(nb_factors_asset)]
                                 for factor_2 in range(nb_factors_asset)])
                    if self.model_skew_ln_ind == 'ln_ln':
                        sim_curr = self.simulated_curves[asset_nb][(t_i != 0) * (t_i-1), tenor_idx, :] + \
                            delta_X - 0.5 * qv
                        self.simulated_curves[asset_nb][t_i, tenor_idx, :] = sim_curr
                    else:  # skew model, qv differently computed
                        X_prev[asset_nb][tenor_idx, :] = X[asset_nb][tenor_idx, :]
                        X[asset_nb][tenor_idx, :] = X_prev[asset_nb][tenor_idx, :] + delta_X
                        c1 = self.C_vec_list[asset_nb][tenor_nb, 0]
                        c2 = self.C_vec_list[asset_nb][tenor_nb, 1]
                        c3 = self.C_vec_list[asset_nb][tenor_nb, 2]
                        t_prev = 0.
                        t_next = self.simulation_times[t_i]
                        # V used
                        V_u = np.sum([[self._V_cross_factor(asset_nb, factor_1, factor_2,
                                                            tenor_nb, tenor_nb, t_prev, t_next)
                                      for factor_1 in range(nb_factors_asset)]
                                      for factor_2 in range(nb_factors_asset)])
                        X_u = X[asset_nb][tenor_idx, :]  # delta_X # X used
                        F_u = self.forward_curve_list[asset_nb][tenor_nb]
                        F_res = F_u * (1. + X_u + 0.5 * c1 * (X_u**2 - V_u) +
                                       c2 * (X_u**3 - 3. * X_u * V_u) / 6. +
                                       c3 * (X_u**4 - 6. * V_u * X_u**2 + 3. * V_u**2) / 24.)
                        self.simulated_curves[asset_nb][t_i, tenor_idx, :] = F_res

        if self.model_skew_ln_ind == 'ln_ln':
            for asset_nb in range(self.nb_assets):
                self.simulated_curves[asset_nb] = pycuda.cumath.exp(self.simulated_curves[asset_nb])
