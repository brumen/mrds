import config  # general configuration
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats
import scipy.optimize
import scipy.linalg
import multiprocessing as mp
import copy
if config.CUDA_PRESENT:
    import pycuda.autoinit
    import pycuda.gpuarray as gpa
    import pycuda.cumath
    import cuda_ops as co

# abstract classes
import ds
import sg
import pricers_fast  # fast libs in cython
import vols


def cdf_vec(x, ci=False):
    """
    computes the cdf of either x which is on the host or the device
    on 100.000 elts, the gpu is approximately 10x faster than cpu
    """
    if not ci:
        return cdf_vec_cpu(x)
    else:
        return co.cdf_vec_gpu(x)


def cdf_vec_cpu(x):
    """
    consult the function cdf in pricers_fast.pyx
    works for both vectors and matrices
    :param ci: cuda indicator
    """
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429

    l = np.abs(np.array(x))
    k = 1. / (1. + 0.2316419 * l)
    k2 = k**2
    k4 = k2**2
    # 0.39 = 1/sqrt(2*pi)
    w = 1. - 0.3989422804 * np.exp(-l*l / 2) * (a1 * k + a2 * k2 +
                                                a3 * k2 * k + a4 * k4 + a5 * k4 * k)

    return w * (x >= 0.) + (1. - w) * (x < 0.)


def pdf_vec(x):
    return np.exp(-x**2/2.)/np.sqrt(2. * np.pi)


def bvnd(dh, dk, r):
    """
    bivariate distribution function at (dh, dk) for standard normals, correlated with r
    """
    return bvnu(-dh, -dk, r)


def bvnu(dh, dk, r):
    """
    2-dimensional distribution function, from
        Alan Genz
        described in "On the computation of the bivariate normal integral",
        Journal of Statist. Comput. Simul. 35, pp 101-107
    """
    tp = 2. * np.pi
    h = dh
    k = dk
    hk = h*k
    bvn = 0.

    if np.abs(r) < 0.3:  # column vectors (same below)
        w = np.array([[0.1713244923791705],
                      [0.3607615730481384],
                      [0.4679139345726904]])
        x = np.array([[0.9324695142031522],
                      [0.6612093864662647],
                      [0.2386191860831970]])
    elif np.abs(r) < 0.75:  # Gauss Legendre points and weights, n = 12
        w = np.array([[.04717533638651177], [0.1069393259953183], [0.1600783285433464],
                      [0.2031674267230659], [0.2334925365383547], [0.2491470458134029]])
        x = np.array([[0.9815606342467191], [0.9041172563704750], [0.7699026741943050],
                      [0.5873179542866171], [0.3678314989981802], [0.1252334085114692]])
    else:  # Gauss-Legendre n = 20
        w = np.array([[.01761400713915212], [.04060142980038694], [.06267204833410906],
                      [.08327674157670475], [0.1019301198172404], [0.1181945319615184],
                      [0.1316886384491766], [0.1420961093183821], [0.1491729864726037],
                      [0.1527533871307259]])
        x = np.array([[0.9931285991850949], [0.9639719272779138], [0.9122344282513259],
                      [0.8391169718222188], [0.7463319064601508], [0.6360536807265150],
                      [0.5108670019508271], [0.3737060887154196], [0.2277858511416451],
                      [0.07652652113349733]])

    w = np.concatenate((w, w))
    x = np.concatenate((1. - x, 1. + x))  # vector
    if np.abs(r) < 0.925:
        hs = 0.5 * (h**2 + k**2)  # / 2.  # number or row vector
        asr = 0.5 * np.arcsin(r)  # / 2.  # number
        sn = np.sin(asr * x)  # col. vector
        bvn = np.sum(np.exp((sn * hk - hs)/(1. - sn**2))*w, axis=0)  # number (vector * vector)
        bvn = bvn * asr / tp + cdf_vec(-h) * cdf_vec(-k)  # number
    else:
        if r < 0.:
            k = -k
            hk = - hk
        if np.abs(r) < 1.:
            as_n = 1. - r**2
            a = np.sqrt(as_n)
            bs = (h-k)**2
            asr = - (bs/as_n + hk)/2.
            c = (4. - hk)/8.
            d = (12. - hk)/80.
            if asr > 100.:
                bvn = a * np.exp(asr) * (1. - c*(bs-as_n)*(1.-d*bs)/3. + c*d*as_n**2)
            if hk > -100.:
                b = np.sqrt(bs)
                sp = np.sqrt(tp) * cdf_vec(-b/a)
                bvn -= np.exp(-hk/2.) * sp * b * (1. - c * bs * (1. - d * bs)/3.)

            a /= 2.
            xs = (a*x)**2
            asr = -0.5 * (bs/xs + hk)
            ix = asr > -100.
            sp = 1. + c * xs * (1. + 5. * d * xs)
            rs = np.sqrt(1. - xs)
            ep = np.exp(-hk/2. * xs / (1. + rs)**2) / rs
            bvn = (a * np.sum(np.exp(asr)*ix * (sp-ep)*w, axis=0) - bvn) / tp

        if r > 0.:
            bvn += cdf_vec(- np.maximum(h, k))
        elif h >= k:
            bvn = - bvn
        else:
            if h < 0.:
                L = cdf_vec(k) - cdf_vec(h)
            else:
                L = cdf_vec(-h) - cdf_vec(-k)
            bvn = L - bvn

    return np.maximum(0., np.minimum(1., bvn))


def apo_long_f(args):
    """
    multiprocessing function for APO long option
    APO based on moment matching
    """
    sim, F_c_mat, apo_c_K, df, swap_ext_mat, \
        sigma_c, rho_mat, forward_tenors, option_tenors, \
        t, beta, sigma_L, cp_ind = args

    return pricers_fast.apo_long_fast(F_c_mat[:, sim], apo_c_K, df, swap_ext_mat,
                                      sigma_c, rho_mat,
                                      forward_tenors, option_tenors,
                                      t, beta, sigma_L, cp_ind)


def black_greeks(S_0, K, r, sigma, T, cp_ind='c',
                 price_only=False,
                 fast_appx=True):
    """
    Black's formula implementation
    the return arguments are
      Black ... value of the option
      Delta, Gamma, Vega, Theta, Rho
      call_put_ind == c for CALL
                      p for PUT
                      b for BINARY
    """
    sqrtt = np.sqrt(T)
    d1 = (np.log(S_0/K) + 0.5 * sigma**2 * T)/(sigma*sqrtt)
    d2 = d1 - sigma * sqrtt
    if not fast_appx:
        n11 = scipy.stats.norm.cdf(d1)
        n22 = scipy.stats.norm.cdf(d2)
    else:
        n11 = cdf_vec(d1)
        n22 = cdf_vec(d2)
    disc = np.exp(- r * T)
    black = disc * (S_0 * n11 - K * n22)
    if price_only:
        if cp_ind == 'c':
            return black
        else:
            return black + (K - S_0) * disc
    else:
        n1 = scipy.stats.norm.pdf(d1)
        if cp_ind == 'c':
            delta = disc * n11
            gamma = disc * n1 / (S_0 * sigma * sqrtt)
            vega = disc * S_0 * n1 * sqrtt
            theta = -S_0 * disc * n1 * sigma / (2. * sqrtt) + \
                r * S_0 * disc * n11 - r * K * disc * n22
            rho = K * T * disc * n22
        else:
            delta = disc * (n11 - 1.)
            gamma = disc * n1 / (S_0 * sigma * sqrtt)
            vega = disc * S_0 * n1 * sqrtt
            theta = -S_0 * n1 * sigma / (2 * sqrtt) + \
                r * K * disc * scipy.stats.norm.cdf(-d2) - r * S_0 * disc * scipy.stats.norm.cdf(-d1)
            rho = - K * T * disc * scipy.stats.norm.cdf(-d2)
        return np.array([black, delta, gamma, vega, theta, rho])


def black_simple(date_, com, exp_, strike, quant=1.):
    """
    simple version of the black_simple
    quant ... quantity
    """
    T = ds.time_diff(date_, exp_)
    fv = ds.read_data_matched_tenors(date_, com, com)
    tenor_v, F_v = fv['fwd_tenors_dt'], fv['fwd_curve']
    opt_ten_v, sigma_v = fv['option_tenors_dt'], fv['vol_surface_params']
    tenor_idx = np.sum([t < ds.convert_str_datetime(exp_)
                        for t in tenor_v])
    S0 = F_v[tenor_idx]
    vol_type = str(ds.vol_hash[com])
    if vol_type == 'JWSS7':
        vol_o = vols.jw7_params(date_, com, com)
        sigma_u = vol_o.implied_vol(tenor_idx, strike, T)
    elif vol_type == 'ATM_RR_WG':
        vol_o = vols.ci7_param(date_, com, com)
        sigma_u = vol_o.implied_vol_strike[tenor_idx](strike)
    elif vol_type == 'ATM':
        sigma_u = sigma_v[tenor_idx]

    df = ds.DF(date_, exp_)
    r = -np.log(df)/T

    return quant * black_greeks(S0, strike, r, sigma_u, T)


def trivariate_spread_kirk(F_v, K, sigma_v, rho, T, DF,
                           lu_int_b=[-20., 20.]):
    """
    trivariate spread option based on Kirk formula
      F_v = [F_1, F_2, F_3]
      sigma_v is a vector of sigmas [sigma_1, sigma_2, sigma_3]
      rho_v = [rho_12, rho_13, rho_23]
    """
    nu = sigma_v * np.sqrt(T)
    mu = - 0.5 * nu ** 2  # + np.log (F_v)
    nu_1_d = nu[0] * np.sqrt(1. - rho[1]**2)
    nu_2_d = nu[1] * np.sqrt(1. - rho[2]**2)
    mu_1_d = lambda x_3: mu[0] + rho[1] * x_3 * nu[0]
    mu_2_d = lambda x_3: mu[1] + rho[2] * x_3 * nu[1]
    k_1 = lambda x_3: K + F_v[2] * np.exp(x_3 * nu[2] + mu[2])
    rho_Y1_Y2 = (rho[0] - rho[1]*rho[2])/np.sqrt(1.-rho[1]**2) / np.sqrt(1. - rho[2]**2)
    
    kirk_integ = lambda x_3: spread_option_kirk(F_v[0] * np.exp(mu_1_d(x_3) + 0.5 * nu_1_d**2),
                                                F_v[1] * np.exp(mu_2_d(x_3) + 0.5 * nu_2_d**2),
                                                k_1(x_3),
                                                nu_1_d/np.sqrt(T),
                                                nu_2_d/np.sqrt(T),
                                                rho_Y1_Y2, T, DF) / \
        np.sqrt(2. * np.pi) * np.exp(-x_3**2/2.)

    return scipy.integrate.quad(kirk_integ, lu_int_b[0], lu_int_b[1])[0]


def trivariate_spread_exact(F_v, K, sigma_v, rho, T, DF,
                            quad_spgrid_ind='quad',
                            sg_level=10):
    """
    exact version of trivariate spread option, based on sparse grids, or scipy integration
    """
    nu = sigma_v * np.sqrt(T)
    mu = - 0.5 * nu ** 2  # + np.log (F_v)

    nu_1_d = nu[0] * np.sqrt(1. - rho[1]**2)
    nu_2_d = nu[1] * np.sqrt(1. - rho[2]**2)
    mu_1_d = lambda x_3: mu[0] + rho[1] * x_3 * nu[0]
    mu_2_d = lambda x_3: mu[1] + rho[2] * x_3 * nu[1]
    k_1 = lambda x_3: K + F_v[2] * np.exp(x_3 * nu[2] + mu[2])
    rho_Y1_Y2 = (rho[0] - rho[1]*rho[2])/np.sqrt(1-rho[1]**2) / np.sqrt(1. - rho[2]**2)
    # sigma_Z = np.sqrt(1. - rho_Y1_Y2**2)
    eta = lambda x_3, y_2: mu_1_d(x_3) + rho_Y1_Y2 * y_2 * nu_1_d + 0.5 * nu_1_d ** 2 * (1. - rho_Y1_Y2**2)
    K_2 = lambda x_3, y_2: k_1(x_3) + F_v[1] * np.exp(y_2 * nu_2_d + mu_2_d(x_3))

    kirk_integ = lambda x_3, y_2: black_greeks(F_v[0] * np.exp(eta(x_3, y_2)), K_2(x_3, y_2),
                                               - np.log(DF)/T,
                                               nu_1_d * np.sqrt(1. - rho_Y1_Y2**2)/np.sqrt(T), T, 0)[0]
    #                 / (2. * np.pi) * np.exp(-(x_3**2 + y_2 **2)/2.0)

    if quad_spgrid_ind == 'quad':  # in-built quadrature fct.
        return scipy.integrate.dblquad(lambda x, y: kirk_integ(x, y) * np.exp(-(x**2 + y**2)/2.), -np.inf, np.inf,
                                       lambda x: -np.inf, lambda x: np.inf,
                                       epsabs=1e-2, epsrel=1e-2)[0] / (2 * np.pi)
    else:  # sparse grid ind
        return sg.sg_quad(2, sg_level, lambda xy: kirk_integ(xy[0], xy[1]))


def trivariate_spread_exact_fast(F_v, K, sigma_v, rho, T, DF):
    """
    same as above except that it uses the cython version of trivariate spread function (great improvement)
    """
    return scipy.integrate.dblquad(lambda x_3, y_2:
                                   pricers_fast.trivariate_spread_exact_integrat_fast2(x_3, y_2,
                                                                                       F_v, K,
                                                                                       sigma_v, rho, T, DF),
                                   -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]


def multivariate_spread_mm(multi_option_fct, l, K, sim_t_i, T, mm, fwd_idx):
    """
    computes the spread F_1 - F_2 - F_3 - ... F_K for multiple repetitions of the F, which are given as
    distinct columns in the matrix
    mm is the market model containing values:
      nb_assets
      simulated_curves (standard form, refer to mrds doc)
      V_fct_current
      market_corr_list
      discount function
      simulation_times
    tri_option_fct ... function that computes the value of the trivariate option
    """
    # reordering of the simulated paths here
    multi_nb = len (l) 
    if multi_nb != mm.nb_assets:
        print "Multiplier vector does not equal the number of assets."
        return -1  # return the error message -1
    
    F_v_mat = np.kron(l.reshape (multi_nb,1), np.ones(mm.simulated_curves[0].shape[2])) * \
              np.array([mm.simulated_curves[asset][sim_t_i, fwd_idx, :]
                        for asset in range(mm.nb_assets)])
    sigma_v = np.array([np.sqrt(mm.V_fct_current (asset_nb, fwd_idx, T) / T )
                        for asset_nb in range(mm.nb_assets)]) # !!! WRONG WRONG WRONG
    rho = [mm.market_corr_list[i][j][fwd_idx]
           for i in range(mm.nb_assets)
           for j in range(i+1,mm.nb_assets)]
    DF = scipy.interpolate.splev(T, mm.discount_function) / \
         scipy.interpolate.splev(mm.simulation_times[sim_t_i], mm.discount_function)
    tv_i = lambda i: multi_option_fct(F_v_mat[:, i], K, sigma_v, rho, T, DF)

    return np.array([tv_i(i) for i in range(F_v_mat.shape[1])])


def spread_option_kirk(F_1, F_2, K, sigma_1, sigma_2, rho, T, DF):
    """
    kirk formula for bivariate spread option when strike K = 0
    """
    sigma_K = np.sqrt(sigma_1**2 - 2 * F_2 / (F_2 + K) * rho * sigma_1 * sigma_2 +
                      (F_2 / (F_2 + K)) ** 2 * sigma_2**2)
    d_1 = (np.log(F_1/(F_2+K)) + 0.5 * sigma_K**2 * T) / (sigma_K * np.sqrt(T))
    d_2 = d_1 - sigma_K * np.sqrt(T)

    return  DF * (F_1 * cdf_vec(d_1) - (F_2+K) * cdf_vec(d_2))

# these vars. are needed in d11 function
p_integ1 = np.array([-3.66847085e+00,  -2.78329010e+00,   3.66847085e+00,
                     -2.02594802e+00,  -1.32655708e+00,  -6.56809567e-01,
                     -1.06611759e-16,   6.56809567e-01,   1.32655708e+00,
                     2.78329010e+00,   2.02594802e+00])
p_integ = p_integ1.reshape(11, 1)
w_integ1 = np.array([1.43956039e-06,   3.46819466e-04,   1.43956039e-06,
                    1.19113954e-02,   1.17227875e-01,   4.29359752e-01,
                    6.54759287e-01,   4.29359752e-01,   1.17227875e-01,
                    3.46819466e-04,   1.19113954e-02])
w_integ = w_integ1.reshape(11, 1)


def d11_helper(params, z):
    """ 
    returns the value of the spread option 
    """
    S_1, S_2, K, sigma_1, sigma_2, rho, T, DF = params
  
    # accessory variables
    nu_1 = sigma_1 * np.sqrt(T)
    nu_2 = sigma_2 * np.sqrt(T)
    mu_1 = np.log(S_1) - 0.5 * sigma_1 **2 * T
    mu_2 = np.log(S_2) - 0.5 * sigma_2 **2 * T
    m_over = rho * z
    s_sqr = 1. - rho **2
    s_sqrt = np.sqrt(s_sqr)
    V = (np.log(np.exp(z * nu_2 + mu_2) + K) - mu_1)/nu_1

    cdf_int_1 = (m_over - V)/s_sqrt + s_sqrt*nu_1
    cdf_int_2 = cdf_int_1 - s_sqrt*nu_1
    H_1 = np.exp(mu_1 + m_over * nu_1 + 0.5 * s_sqr * nu_1 * nu_1) * cdf_vec(cdf_int_1)
    H_2 = (np.exp(z * nu_2 + mu_2) + K) * cdf_vec(cdf_int_2)

    return H_1 - H_2


def spread_option_appx(F_1, F_2, K, sigma_1, sigma_2, rho, T, DF):
    """
    appx valuation of spread option 
    """
    params = [F_1, F_2, K, sigma_1, sigma_2, rho, T, DF]
    f_vals = d11_helper(params,  p_integ * np.sqrt(2.))
    return DF * np.sum(w_integ * f_vals, axis=0) / np.sqrt(np.pi)


def spread_option_exact(F_1, F_2, K, sigma_1, sigma_2, rho, T, DF):
    params = [F_1, F_2, K, sigma_1, sigma_2, rho, T, DF]
    f_integ = lambda z: d11_helper(params, z) * \
        np.exp(-z**2/2.) / np.sqrt(2. * np.pi)
    return DF * scipy.integrate.quad(f_integ, -np.inf, np.inf)[0]


def spread_option_krik_zero_strike(F_1, F_2, sigma_1, sigma_2, rho, T, DF):
    return spread_option_kirk(F_1, F_2, 0., sigma_1, sigma_2, rho, T, DF)


def apo_wo_basket (K, sim_t_i, T, mm, fwd_idx ):
    """
    implements APO WO BASKET option with strike K for a fixed fwd_idx
    """
    avg = np.array([np.mean(mm.simulated_curves[asset_nb][:, fwd_idx, :], 1)
                    for asset_nb in range(mm.nb_assets)])
    # minimum on columns (columns are assets)
    return np.mean(np.array([np.min(avg[:, sim]) - K
                             for sim in range(mm.nb_simulations)]))


def apo_long(F_c, K, df, T, sigma_c, rho_mat, Ti_c, ti_c, t, beta, sigma_L,
             call_put_ind='call'):
    """
    long APO calculation
    :param call_put_ind: 0 for CALL
                         1 for PUT
    :param F_c: futures curve
    :param sigma_c: volatility curve
    :param rho_mat: correlation between individual futures' contracts CHECK THIS CHECK THIS...
    :param Ti_c, ti_c: future's maturity, option maturity for a futures' contract
    :param beta, sigma_L: Samuelson params
    :param K: apo strike price (CHECK THIS CHECK THIS)
    :param df: discount factor until the APO maturity  CHECK CHECK CHECK
    :param T: APO maturity CHECK CHECK CHECK
    """
    N = len(F_c)

    def A(T_a, T_b, t, t_i):
        t1 = sigma_L**2 * (t_i-t)
        t2 = sigma_L * (np.exp(-beta * (T_b - t_i)) - np.exp(-beta * (T_b - t)) +
                        np.exp(-beta * (T_a - t_i)) - np.exp(-beta * (T_a - t))) / beta
        t3 = (np.exp(-2 * beta * ((T_a + T_b) / 2. - t_i)) - np.exp(-2 * beta * ((T_a + T_b) / 2. - t))) / (2*beta)
        return (t1 + t2 + t3) / ((t_i - t) * vols.sam_int(t, T_a, T_a, beta, sigma_L) *
                                 vols.sam_int(t, T_b, T_b, beta, sigma_L))

    M_1 = np.mean(F_c)
    M_2_term1 = F_c**2 * np.array([np.exp(A(Ti_c[i], Ti_c[i], t, Ti_c[i]) *
                                          sigma_c[i]**2 * (ti_c[i] - t))
                                   for i in range(N)])

    M_2_matrix_term = np.array([2 * F_c[i] * F_c[j] *
                               np.exp(A(Ti_c[i], Ti_c[j], t, np.min(Ti_c[i], Ti_c[j])) *
                                      rho_mat[i][j] * sigma_c[i] * sigma_c[j] * (ti_c[i] - t))
                               for i in range(N) for j in range(i+1, N)])
    M_2 = (np.sum(M_2_term1) + np.sum(M_2_matrix_term)) / (np.double(N**2))
    sigma = np.sqrt(np.log(M_2/M_1**2))/np.sqrt(T)

    return black_greeks(M_1, K, -np.log(df)/T, sigma, T, call_put_ind != 'call')


def apo_vector(F_c_mat, apo_c_K, df, maturity,
               sigma_c_fwd, rho_mat,
               forward_tenors, option_tenors,
               t, beta, sigma_L, cp_ind, nb_sims, nb_sims_switch=10000):
    """
    wrapper function for apo call/put
    constructs a vector of apo prices for different F_c (columns of F_c_mat)
    :param nb_sims_switch: at what number of simulations should it switch to the
    """
    if nb_sims < nb_sims_switch:
        return np.array([pricers_fast.apo_long_fast(F_c_mat[:, sim], apo_c_K, df, maturity,
                                                    sigma_c_fwd, rho_mat,
                                                    forward_tenors, option_tenors,
                                                    t, beta, sigma_L, cp_ind)
                         for sim in range(nb_sims)])

    else:  # nb_sims >= nb_sims_switch
        nb_cores = mp.cpu_count()
        pool = mp.Pool(processes=nb_cores)
        # WRONG WRONG WRONG: EXT_TENORS MISSSING
        return np.array(pool.map(apo_long_f, zip(range(nb_sims), [F_c_mat] * nb_sims,
                                                 [apo_c_K] * nb_sims, [df] * nb_sims,
                                                 [maturity] * nb_sims,
                                                 [sigma_c_fwd] * nb_sims, [rho_mat] * nb_sims,
                                                 [forward_tenors] * nb_sims,
                                                 [option_tenors] * nb_sims,
                                                 [t] * nb_sims,
                                                 [beta] * nb_sims, [sigma_L] * nb_sims,
                                                 [cp_ind]*nb_sims )))


def cont_extend(mm, params):
    """
    contingent extendible pricing (extendible is a cont. ext with _one_ extension
      period) - only multiple extensions currently allowed
    forward curve simulated only up to 1 point
    params - list of
     [ ext_mat, swap_mat] ... maturity of the extendable and swap maturity (> ext_mat)
     [ swap_K, ...] ... strikes of swap, APO call and puts
     [ swap_nb ,...] ... amount of swaps, APO calls, and puts
     [ beta, sigma_L ] ... samuelson params
     sim_ind ... index of the mm.simulated_curves[0][sim_ind, appr. index, :] to take
     fwd_corr ... correlation of the forward contracts, assumed constant
    """
    ext_mat_v = params[0][0]
    swap_mat_v = params[0][1] # these two vectors have to be of the same length
    # APO call/put strikes 
    swap_K_v = params[1][0]
    apo_c_K_v = params[1][1]
    apo_p_K_v = params[1][2]
    # APO call/put numbers 
    swap_nb_v = params[2][0]
    apo_c_nb_v = params[2][1]
    apo_p_nb_v = params[2][2]
    # samuelson params
    beta = params[3][0]
    sigma_L = params[3][1]
    # additional params 
    sim_ind = params[4]
    fwd_corr = params[5] 
    
    nb_sims = shape (mm.simulated_curves[0])[2]
    nb_ext = len (ext_mat_v) 
    
    # extract monthly indices between ext_mat and swap_mat for _first_ extension period 
    ext_tenors = range ( sum (mm.forward_tenors_list[0] < ext_mat_v[0] ),
                         sum (mm.forward_tenors_list[0] < swap_mat_v[0] ) ) 
    
    F_c_mat = mm.simulated_curves[0][sim_ind, ext_tenors, :] # the extendible maturiy time
    sigma_c = mm.atm_vol_list[0][ext_tenors] # WRONG WRONG, NEEDS CORRECTION CORRECTION CORRECTION 
    df = mm.DF(swap_mat_v[0]) / mm.DF(ext_mat_v[0])  # discount from beginning to the end
    dfs = mm.DF(mm.forward_tenors_list[0][ext_tenors]) / mm.DF (ext_mat_v[0] ) # disc. factors until payment days
    
    rho_mat = vols.corr_hyp_sec_mat (fwd_corr, range(len (ext_tenors)))
    t = 0.  # THIS SHOULD BE REMOVED REMOVED REMOVED 
    
    sigma_c_fwd = vols.forward_vols_sam(sigma_c, ext_mat_v[0], mm.forward_tenors_list[0][ext_tenors],
                                        mm.option_tenors_list[0][ext_tenors], beta, sigma_L)

    # skew adjustment of K (replaces sigma_c_fwd above)
    call_K_mat_skew = pricers_fast.comp_skew_strikes (mm.simulated_curves[0][sim_ind, :, :], np.array(ext_tenors),
                                                      mm.option_tenors_list[0], ext_mat_v[0], apo_c_K_v[0],
                                                      beta, sigma_L)

    delta_fwd = np.log(F_c_mat / call_K_mat_skew[ext_tenors, :]) / sigma_c.reshape((len(ext_tenors), 1))  # TENORS MISSING TENORS MISSING

    sigma_skew_fwd = np.array([np.diag(mm.vol_surface_function[0](mm.option_tenors_list[0][ext_tenors],
                                                                  delta_fwd[:, sim]))
                               for sim in range (nb_sims)])

    sigma_c_fwd = vols.forward_vols_sam(sigma_skew_fwd, ext_mat_v[0], mm.forward_tenors_list[0][ext_tenors],
                                        mm.option_tenors_list[0][ext_tenors], beta, sigma_L)

    # first extension 
    # THIS EXTENSION CAN BE REWRITTEN 
    nb_sims_switch = 100000
    # apo call value computing 
    if apo_c_nb_v[0] != 0:
        apo_c_v = apo_vector(F_c_mat, apo_c_K_v[0], df, swap_mat_v[0] - ext_mat_v[0],
                             sigma_c_fwd, rho_mat,
                             mm.forward_tenors_list[0][ext_tenors],
                             mm.option_tenors_list[0][ext_tenors],
                             t, beta, sigma_L, 1., nb_sims, nb_sims_switch)
    else:
        apo_c_v = 0.

    # apo put value computing 
    if apo_p_nb_v[0] != 0.:
        apo_p_v = apo_vector (F_c_mat, apo_c_K_v[0], df, swap_mat_v[0] - ext_mat_v[0],
                              sigma_c_fwd, rho_mat,
                              mm.forward_tenors_list[0][ext_tenors],
                              mm.option_tenors_list[0][ext_tenors],
                              t, beta, sigma_L, 0., nb_sims, nb_sims_switch)
    else:
        apo_p_v = 0.

    # swap value computation 
    disc_mat = dfs.reshape(len(dfs), 1)
    swap_v = np.mean((F_c_mat - swap_K_v[0]) * disc_mat, axis=0)

    if nb_ext == 1: 
        portf = apo_c_nb_v[0] * apo_c_v + apo_p_nb_v[0] * apo_p_v + swap_nb_v[0] * swap_v
    else: 
        # further extensions (change of parameters) 
        params_1 = [[ext_mat_v[1:nb_ext], swap_mat_v[1:nb_ext]],
                    [swap_K_v[1:nb_ext], apo_c_K_v[1:nb_ext], apo_p_K_v[1:nb_ext]],
                    [swap_nb_v[1:nb_ext], apo_c_nb_v[1:nb_ext], apo_p_nb_v[1:nb_ext]],
                    [beta, sigma_L],
                    sim_ind, fwd_corr]

        # overwrite the market model for further extensions
        ce = np.zeros(nb_sims)
        mme = copy.deepcopy(mm)  # copy of the market object mm
        for sim_nb in range(nb_sims):
            mme.forward_curve_list[0] = mm.simulated_curves[0][sum (mm.forward_tenors_list[0] <ext_mat_v[0]) -1,:,sim_nb] 
            sim_times = (ext_mat_v[1] - ext_mat_v[0]) * np.arange(5) / 5.
            # THIS IS NOT COMPLETELY CORRECT NOT CORRECT, FORWARD_TENOR_LIST NOT UPDATED CORRECTLY
            # + FORWARD VOLS IGNORED
            mme.update_sim_times (sim_times) 
            mme.simulate_curves(5000)
            ce[sim_nb] = cont_extend(mme, params_1)
            print "Path ", sim_nb, ". Res = ", ce[sim_nb]

        portf = apo_c_nb_v[0] * apo_c_v + apo_p_nb_v[0] * apo_p_v + swap_nb_v[0] * swap_v + ce
    return mm.DF(ext_mat_v[0]) * np.mean(portf * (portf > 0))


def spread_extend(mm, params):
    """
    spread extendible pricing
    forward curve simulated only up to 1 point
    :param [ext_mat, swap_mat]: maturity of the extendable and swap maturity (> ext_mat)
    :param [swap_K, ...]: strikes of swap, APO call and puts
    :param [swap_nb ,...]: amount of swaps, APO calls, and puts
     [ beta, sigma_L ] ... samuelson params
     sim_ind ... index of the mm.simulated_curves[0][sim_ind, appr. index, :] to take
    """
    ext_mat = params[0][0]
    swap_mat = params[0][1]
    # APO call/put strikes 
    swap_K = params[1][0]
    apo_c_K = params[1][1]
    apo_p_K = params[1][2]
    # APO call/put numbers 
    swap_nb = params[2][0]
    apo_c_nb = params[2][1]
    apo_p_nb = params[2][2]
    # samuelson params
    beta = params[3][0]
    sigma_L = params[3][1]
    # additional params 
    sim_ind = params[4] 
    fwd_corr_1 = params[5][0]
    fwd_corr_2 = params[5][1]

    nb_sims = np.shape(mm.simulated_curves[0])[2]
    
    # extract monthly indices between ext_mat and swap_mat
    ext_tenors = range(np.sum(mm.forward_tenors_list[0] < ext_mat),
                       np.sum(mm.forward_tenors_list[0] < swap_mat))

    F_c0_mat = mm.simulated_curves[0][sim_ind, ext_tenors, :]  # the extendible maturiy time of asset 0
    F_c1_mat = mm.simulated_curves[1][sim_ind, ext_tenors, :]  # the extendible maturiy time of asset 1
    sigma_c0 = mm.atm_vol_list[0][ext_tenors]  # WRONG WRONG, NEEDS CORRECTION CORRECTION CORRECTION
    sigma_c1 = mm.atm_vol_list[1][ext_tenors]  # WRONG WRONG, NEEDS CORRECTION CORRECTION CORRECTION
    df = mm.DF(swap_mat) / mm.DF(ext_mat)  # discount from beginning to the end
    dfs = mm.DF(mm.forward_tenors_list[0][ext_tenors]) / mm.DF(ext_mat)  # disc. factors until payment days

    rho_mat_0 = vols.corr_hyp_sec_mat (fwd_corr_1, range(len (ext_tenors))) # correlation for asset 0
    rho_mat_1 = vols.corr_hyp_sec_mat (fwd_corr_2, range(len (ext_tenors))) # correlation for asset 1

    t = 0.  # THIS SHOULD BE REMOVED REMOVED REMOVED 

    sigma_c0_fwd = vols.forward_vols_sam(sigma_c0, ext_mat, mm.forward_tenors_list[0][ext_tenors], mm.option_tenors_list[0][ext_tenors], beta, sigma_L)
    sigma_c1_fwd = vols.forward_vols_sam(sigma_c1, ext_mat, mm.forward_tenors_list[1][ext_tenors], mm.option_tenors_list[1][ext_tenors], beta, sigma_L)

    # apo call value computing 
    if (apo_c_nb != 0) and (nb_sims < 100000) :
        apo_c_v = np.array([ pricers_fast.apo_long_fast (F_c0_mat[:,sim] - F_c1_mat[:,sim], apo_c_K, df,
                                                         swap_mat - ext_mat,
                                                         sigma_c_fwd, rho_mat,
                                                         mm.forward_tenors_list[0][ext_tenors],
                                                         mm.option_tenors_list[0][ext_tenors],
                                                         t, beta, sigma_L, 1.)
                             for sim in range(nb_sims)])

    elif (apo_c_nb != 0.) and (nb_sims >= 100000):
        # multiprocessing
        nb_cores = mp.cpu_count()
        pool = mp.Pool(processes=nb_cores)
        apo_c_v = np.array(pool.map(apo_long_f, zip(range (nb_sims), [mm] * nb_sims, [F_c0_mat - F_c1_mat] * nb_sims,
                                                    [apo_c_K] * nb_sims, [df] * nb_sims,
                                                    [swap_mat - ext_mat] * nb_sims,
                                                    [sigma_c] * nb_sims, [rho_mat] * nb_sims,
                                                    [mm.forward_tenors_list[0][ext_tenors]] * nb_sims,
                                                    [mm.option_tenors_list[0][ext_tenors]]* nb_sims,
                                                    [t]* nb_sims, [beta]*nb_sims, [sigma_L]*nb_sims, 1.)))
    else:
        apo_c_v = 0.

    # apo put value computing 
    if (apo_p_nb != 0.) and (nb_sims < 100000):
        apo_p_v = np.array([pricers_fast.apo_long_fast(F_c0_mat[:,sim] - F_c1_mat[:,sim], apo_p_K, df,
                                                       swap_mat - ext_mat,
                                                       sigma_c, rho_mat,
                                                       mm.forward_tenors_list[0][ext_tenors],
                                                       mm.option_tenors_list[0][ext_tenors],
                                                       t, beta, sigma_L, 0.0)
                            for sim in range(nb_sims)])
    elif (apo_p_nb != 0.) and (nb_sims >= 100000):  # multiprocessing
        nb_cores = mp.cpu_count()
        pool = mp.Pool(processes=nb_cores)
        apo_p_v = np.array(pool.map(apo_long_f, zip(range(nb_sims), [mm] * nb_sims, [F_c0_mat - F_c1_mat] * nb_sims,
                                                    [apo_c_K] * nb_sims, [df] * nb_sims,
                                                    [swap_mat - ext_mat] * nb_sims,
                                                    [sigma_c] * nb_sims, [rho_mat] * nb_sims,
                                                    [mm.forward_tenors_list[0][ext_tenors]] * nb_sims,
                                                    [mm.option_tenors_list[0][ext_tenors]]* nb_sims,
                                                    [t]* nb_sims, [beta]*nb_sims, [sigma_L]*nb_sims, 0.)))
    else:
        apo_p_v = 0.

    # swap value computation 
    disc_mat = dfs.reshape(len(dfs),1)
    swap_v = np.mean((F_c0_mat - F_c1_mat - swap_K) * disc_mat, axis=0)
    # final portfolio value
    portf = apo_c_nb * apo_c_v + apo_p_nb * apo_p_v + swap_nb * swap_v
    return mm.DF(ext_mat) * np.mean(portf * (portf > 0))


def swap_cva(F_sim, swap_rate, cuda_ind=False):
    if not cuda_ind:
        return np.sum(F_sim - swap_rate, axis=0)
    else:
        return co.colsum_cuda_last(F_sim - swap_rate)
