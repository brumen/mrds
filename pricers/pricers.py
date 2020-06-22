# Pricers file.
import numpy as np
import scipy.integrate
import scipy.stats
import scipy.interpolate
import logging
import QuantLib as ql

from typing import Tuple, Union

from mrds.config import CUDA_PRESENT
if CUDA_PRESENT:
    import pycuda.autoinit
    import cuda.cuda_ops as co

import mrds.sg as sg
import mrds.pricers.pricers_fast as pricers_fast

from mrds.vols.vols_basic import sam_int

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cdf_vec(x, ci=False):
    """ Computes the cdf of either x which is on the host or the device.
           on 100.000 elts, the gpu is approximately 10x faster than cpu

    :param ci: indicator whether to use cuda
    :returns: vector of cdf, if ci = False, then the memory allocation is on CPU, else on GPU
    """

    return cdf_vec_cpu(x) if not ci else co.cdf_vec_gpu(x)


def cdf_vec_cpu(x: np.array) -> np.array:
    """ Computes the cdf of the standard normal random variable of a vector x.
        Works for both vectors and matrices

    :param x: vector/matrix to compute the standard normal variable of.
    :returns: vector/matrix of results

    """

    l  = np.abs(x)
    k  = 1. / (1. + 0.2316419 * l)
    k2 = k**2
    k4 = k2**2

    # 0.39 = 1/sqrt(2*pi)
    w = 1. - 0.3989422804 * np.exp(-l*l / 2) * (0.31938153 * k -0.356563782 * k2 +
                                                1.781477937 * k2 * k + -1.821255978 * k4 + 1.330274429 * k4 * k)

    xPos = x >= 0.
    return w * xPos + (1. - w) * (~xPos)


def pdf_vec(x: np.array) -> np.array :
    """ Standardized normal vector of pdfs.
    """

    return np.exp(-x**2/2.)/np.sqrt(2. * np.pi)


def bvnd(dh, dk, r):
    """
    Bivariate distribution function at (dh, dk) for standard normals, correlated with r

    """

    return bvnu(-dh, -dk, r)


def bvnu(dh, dk, r):
    """
    2-dimensional distribution function, from
        Alan Genz
        described in "On the computation of the bivariate normal integral",
        Journal of Statist. Comput. Simul. 35, power_prices 101-107

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


def black_greeks_local( S_0   : float
                , K     : float
                , r     : float
                , sigma : float
                , T     : float
                , cp_ind     = 'c'
                , price_only = False
                , fast_appx  = True ) -> Union[float, Tuple]:
    """
    Black's formula implementation.

    the return arguments are
      Black ... value of the option
      Delta, Gamma, Vega, Theta, Rho
      call_put_ind == c for CALL
                      network_struct for PUT
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
        return black if cp_ind == 'c' else black + (K - S_0) * disc

    # compute greeks too
    n1 = scipy.stats.norm.pdf(d1)
    delta = disc * n11
    gamma = disc * n1 / (S_0 * sigma * sqrtt)
    vega = disc * S_0 * n1 * sqrtt
    theta = -S_0 * disc * n1 * sigma / (2. * sqrtt) + \
            r * S_0 * disc * n11 - r * K * disc * n22
    rho = K * T * disc * n22

    if cp_ind == 'network_struct':
        # TODO: FINISH HERE!!!
        delta = delta - disc
        theta = -S_0 * n1 * sigma / (2 * sqrtt) + \
              r * K * disc * scipy.stats.norm.cdf(-d2) - r * S_0 * disc * scipy.stats.norm.cdf(-d1)
        rho   = - K * T * disc * scipy.stats.norm.cdf(-d2)  # TODO: THIS CAN BE MADE FASTER.

    return (black, delta, gamma, vega, theta, rho)


def black_quantlib( S_0   : float
                  , K     : float
                  , r     : float
                  , sigma : float
                  , calc_date : ql.Date
                  , maturity  : ql.Date
                  , day_count = ql.ActualActual()
                  , cp_ind     = 'c'
                  , price_only = False ) -> Union[float, Tuple]:

    yield_curve = ql.FlatForward(calc_date, r, day_count, ql.Compounded, ql.Continuous)
    ql.Settings.instance().evaluationDate = calc_date
    flavor = ql.Option.Call if cp_ind == 'c' else ql.Option.Put
    T = yield_curve.dayCounter().yearFraction(calc_date, maturity)
    black = ql.BlackCalculator( ql.PlainVanillaPayoff(flavor, K)
                              , S_0
                              , sigma * np.sqrt(T)
                              , yield_curve.discount(maturity))

    if price_only:
        return black.value()

    return (black.value(), black.delta(), black.gamma(), black.vega(), black.theta(), black.rho())


black_greeks = black_greeks_local


def trivariate_spread_kirk( F       : np.array
                          , K       : float
                          , sigma   : np.array
                          , rho     : np.array
                          , T       : float
                          , DF      : float
                          , lu_int_b=(-20., 20.) ):
    """ Trivariate spread option based on Kirk formula.

    :param F: np.array of forward prices = [F_1, F_2, F_3]
    :param K: strike price
    :param sigma: is a vector of sigmas [sigma_1, sigma_2, sigma_3]
    :param rho: vector of correlations, = [rho_12, rho_13, rho_23]
    :param T: time to maturity
    :param DF: discount factor
    :param lu_int_b: boundary of the integration.
    """

    nu = sigma * np.sqrt(T)
    mu = - 0.5 * nu ** 2  # + np.log (F_v)
    nu_1_d = nu[0] * np.sqrt(1. - rho[1]**2)
    nu_2_d = nu[1] * np.sqrt(1. - rho[2]**2)
    mu_1_d = lambda x_3: mu[0] + rho[1] * x_3 * nu[0]
    mu_2_d = lambda x_3: mu[1] + rho[2] * x_3 * nu[1]
    k_1 = lambda x_3: K + F[2] * np.exp(x_3 * nu[2] + mu[2])
    rho_Y1_Y2 = (rho[0] - rho[1]*rho[2])/np.sqrt(1.-rho[1]**2) / np.sqrt(1. - rho[2]**2)

    kirk_integ = lambda x_3: spread_option_kirk(F[0] * np.exp(mu_1_d(x_3) + 0.5 * nu_1_d**2),
                                                F[1] * np.exp(mu_2_d(x_3) + 0.5 * nu_2_d**2),
                                                k_1(x_3),
                                                nu_1_d/np.sqrt(T),
                                                nu_2_d/np.sqrt(T),
                                                rho_Y1_Y2, T, DF) / \
        np.sqrt(2. * np.pi) * np.exp(-x_3**2/2.)

    lower_bound, upper_bound = lu_int_b
    return scipy.integrate.quad(kirk_integ, lower_bound, upper_bound)[0]


def trivariate_spread_exact( F       : np.array
                           , K       : float
                           , sigma   : np.array
                           , rho     : np.array
                           , T       : float
                           , DF      : float
                           , quad_spgrid_ind = 'quad'
                           , sg_level = 10 ):
    """ Exact version of trivariate spread option, based on sparse grids, or scipy integration

    :param F: np.array of forward prices = [F_1, F_2, F_3]
    :param K: strike price
    :param sigma: is a vector of sigmas [sigma_1, sigma_2, sigma_3]
    :param rho: vector of correlations, = [rho_12, rho_13, rho_23]
    :param T: time to maturity
    :param DF: discount factor
    :param quad_spgrid_ind: indicator whether to use sparse grids or numerical integration ('quad', 'spgrid')
    :param sg_level: level of sparse grids
    """

    nu = sigma * np.sqrt(T)
    mu = - 0.5 * nu ** 2  # + np.log (F_v)

    nu_1_d = nu[0] * np.sqrt(1. - rho[1]**2)
    nu_2_d = nu[1] * np.sqrt(1. - rho[2]**2)
    mu_1_d = lambda x_3: mu[0] + rho[1] * x_3 * nu[0]
    mu_2_d = lambda x_3: mu[1] + rho[2] * x_3 * nu[1]
    k_1 = lambda x_3: K + F[2] * np.exp(x_3 * nu[2] + mu[2])
    rho_Y1_Y2 = (rho[0] - rho[1]*rho[2])/np.sqrt(1-rho[1]**2) / np.sqrt(1. - rho[2]**2)
    # sigma_Z = np.sqrt(1. - rho_Y1_Y2**2)
    eta = lambda x_3, y_2: mu_1_d(x_3) + rho_Y1_Y2 * y_2 * nu_1_d + 0.5 * nu_1_d ** 2 * (1. - rho_Y1_Y2**2)
    K_2 = lambda x_3, y_2: k_1(x_3) + F[1] * np.exp(y_2 * nu_2_d + mu_2_d(x_3))

    kirk_integ = lambda x_3, y_2: black_greeks( F[0] * np.exp(eta(x_3, y_2)), K_2(x_3, y_2)
                                              , - np.log(DF)/T
                                              , nu_1_d * np.sqrt(1. - rho_Y1_Y2**2)/np.sqrt(T), T, 0)[0]
    #                 / (2. * np.pi) * np.exp(-(x_3**2 + y_2 **2)/2.0)

    if quad_spgrid_ind == 'quad':  # in-built quadrature fct.
        return scipy.integrate.dblquad(lambda x, y: kirk_integ(x, y) * np.exp(-(x**2 + y**2)/2.), -np.inf, np.inf,
                                       lambda x: -np.inf, lambda x: np.inf,
                                       epsabs=1e-2, epsrel=1e-2)[0] / (2 * np.pi)
    # sparse grid ind
    return sg.sg_quad(2, sg_level, lambda xy: kirk_integ(xy[0], xy[1]))


def trivariate_spread_exact_fast( F: np.array
                                , K: float
                                , sigma: np.array
                                , rho: np.array
                                , T: float
                                , DF: float
                                ):
    """
    same as above except that it uses the cython version of trivariate spread function (great improvement)
    """
    return scipy.integrate.dblquad(lambda x_3, y_2:
                                   pricers_fast.trivariate_spread_exact_integrat_fast2(x_3, y_2, F, K, sigma, rho, T, DF),
                                   -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]


def multivariate_spread_mm(multi_option_fct, l, K, sim_t_i, T, mm, fwd_idx):
    """
    computes the spread F_1 - F_2 - F_3 - ... F_K for multiple repetitions of the F, which are given as
    distinct columns in the matrix

    mm is the market model containing values:
      nb_assets
      simulated_curves (standard form, refer to mrds doc)
      __V_current
      _market_corr
      discount function
      simulation_times
    tri_option_fct ... function that computes the value of the trivariate option
    """

    # reordering of the simulated paths here
    multi_nb = len (l) 
    if multi_nb != mm.nb_assets:
        logger.info("Multiplier vector does not equal the number of assets.")
        return -1  # return the error message -1

    F_v_mat = np.kron(l.reshape (multi_nb,1), np.ones(mm.simulated_curves[0].shape[2])) * \
              np.array([mm.simulated_curves[asset][sim_t_i, fwd_idx, :]
                        for asset in range(mm.nb_assets)])
    sigma_v = np.array([np.sqrt(mm.__V_current (asset_nb, fwd_idx, T) / T)
                        for asset_nb in range(mm.nb_assets)]) # !!! WRONG WRONG WRONG
    rho = [mm._market_corr[i][j][fwd_idx]
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
    """ returns the value of the spread option
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


def spread_option(F_1, F_2, K, sigma_1, sigma_2, rho, T, DF, exact_appx_ind = 'appx'):
    """
    Appx valuation of spread option, expressed as numerical integration.
    Exact version uses  scipy integration method.

    """

    if exact_appx_ind == 'appx':
        return DF * np.sum( w_integ * d11_helper([F_1, F_2, K, sigma_1, sigma_2, rho, T, DF],  p_integ * np.sqrt(2.))
                          , axis = 0 ) / np.sqrt(np.pi)

    # "exact" method
    return DF * scipy.integrate.quad( lambda z: d11_helper([F_1, F_2, K, sigma_1, sigma_2, rho, T, DF], z) * np.exp(-z**2/2.) / np.sqrt(2. * np.pi)
                                    , -np.inf
                                    , np.inf )[0]


def apo_wo_basket (K, sim_t_i, T, mm, fwd_idx ):
    """
    Implements APO WO BASKET option with strike K for a fixed fwd_idx.

    """

    avg = np.array([np.mean(mm.simulated_curves[asset_nb][:, fwd_idx, :], 1)
                    for asset_nb in range(mm.nb_assets)])
    # minimum on columns (columns are assets)
    return np.mean(np.array([np.min(avg[:, sim]) - K
                             for sim in range(mm.nb_simulations)]))


def apoLong(F_c, K, df, T, sigma_c, rho_mat, Ti_c, ti_c, t, beta, sigma_L,
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
        return (t1 + t2 + t3) / ((t_i - t) * sam_int(t, T_a, T_a, beta, sigma_L) *
                                 sam_int(t, T_b, T_b, beta, sigma_L))

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
