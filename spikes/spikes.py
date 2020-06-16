# spikes model
import numpy as np
import scipy
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats 
import scipy.optimize 
import scipy.interpolate  # spline package
import openopt
import matplotlib
matplotlib.use('TkAgg')

import pricers  # imports the black's pricers, etc.
import spikes_fast
import vols

from mrds import ComSkew


# TODO: REFACTOR ComSkew to fit with this model.
class ComSpikes(ComSkew):

    def __init__(self, mkt_date, lam, delta, eta):
        self.date_ = mkt_date

        # parameters
        self.lam = lam  # frequency of jumps
        self.delta = delta  # length of jump
        self.eta = eta  # jump in prices

    def europe_price(self, S_0, K, sigma, T):
        """
        # price of the european call option in the spike model
        # there exists a fast function called europe_price_spike in spikes_fast module
        """

        d_1 = np.log(S_0/K * (1. + self.eta)) + 0.5*sigma**2 * T
        disc_fact = self.DF(self.date_, T)
        r = - np.log(disc_fact) / T

        return (1. - self.lam * self.delta) * pricers.black_greeks(S_0, K, r, sigma, T, price_only=True) \
            + self.lam * self.delta * pricers.black_greeks(S_0, K/(1. + self.eta), r, sigma, T, price_only=True) + \
            disc_fact * self.lam * self.delta * self.eta * S_0 * scipy.stats.norm.cdf(d_1)

    def europe_price_strip(self, S_0, K, sigma, lam, T_vec):
        # fast value of the spike
        return np.sum(np.array([spikes_fast.europe_price_spike(S_0, K, sigma, T,
                                                               self.eta, lam,
                                                               self.delta, self.DF(T))
                                for T in T_vec])) / np.double(len(T_vec))

    def calib_lambda(self, F300cap, K, S_0, T_vec):
        """
        # calibrates lambda to F300cap prices
        """
        self.lam = openopt.NLP(lambda lam: (self.europe_price_strip(S_0, K, 0.2, lam, T_vec) - F300cap)**2,
                               0.02, lb=1e-8, ub=np.inf).solve('scipy_cobyla').xf

    def impl_vol(self, option_price, S_0, K_v, sigma, T):
        """
        Computes the implied vol, option_price is a function of (S_0, K, sigma, T)


        """

        return vols.black_vol_inverse_vec( S_0
                                         , K_v
                                         , np.array([option_price(S_0, K, sigma, T) for K in K_v])
                                         , T
                                         , self.DF(self.date_, T)
                                         , 1
                                         , 1.e-4 )

    def apo_price(self, S0, K, sigma, T):
        """
        # returns the APO price for spikes model
        """
        def m_fct(y, mu, z):
            # integrating function for m_fct 
            def int1(u):
                if np.isinf(np.cosh(2.*u)):  # exp(- ..) takes care for 0
                    return 0.
                else: 
                    x = scipy.special.hyp1f1(-mu, 1.5, 2. * z * np.sinh(u)**2) * \
                        np.exp(-z * np.cosh(2. * u) - u**2/y) * \
                        np.sinh(2.*u) * np.sin(np.pi * u / y)
                    return 0. if np.isnan(x) else x  # same exp(- ) reasoning

            return 8. * z**(1.5) * scipy.special.gamma(mu + 1.5) * np.exp(np.pi**2 / (4.*y)) / \
                (np.pi * np.sqrt(2 * np.pi * y)) * \
                scipy.integrate.quad(lambda u: int1(u), 0., np.inf)[0]

        def dens(y):
            return 1./(np.sqrt(2.) * y) * np.exp(- sigma**2 * T / 8. - S0/(sigma**2 * y)) * \
                m_fct(sigma**2 * T/2., -1., S0/sigma**2/y)

        def apo_cond(K):
            return scipy.integrate.quad(lambda y: (y-K) * dens(y), max(K, 0.), np.inf)[0]

        # repeat until convergence
        apoc_p = apo_cond(S0 * T)  # term 0
        apoc_p_n = apoc_p * (1. + 2 * 1.e-4)
        delta_rat = (apoc_p - apoc_p_n)/apoc_p  # initial delta_rat
        n = 1  # first term
        while (np.abs(delta_rat) > 1e-4) or (n > 50):
            apoc_p_n = apoc_p 
            apoc_p = apoc_p + apo_cond (S0 * T - self.eta * n) * np.exp(- self.lam * n) * \
                (self.lam * T)**n / scipy.factorial(n)
            n += 1
            delta_rat = (apoc_p - apoc_p_n) / apoc_p

        return apoc_p 

    def calib_par(self, p_v, K_v, S_0, T):
        """
        # calibrates eta, lambda, etc.
        # to fit the market prices
        """
        m_vol_v = vols.black_vol_inverse_vec(S_0, K_v, p_v, T, self.DF(self.date_, T),
                                             0, 1e-4)

        def calib_obj_f(sigma, lam, delta, eta):
            return scipy.linalg.norm(
                self.impl_vol(self.europe_price, S_0, K_v, sigma, T) - m_vol_v)

        self.sigma, self.lam, self.delta, self.eta = \
            openopt.NLP(lambda C: calib_obj_f(C[0], C[1], C[2], C[3]),
                        np.array([0.2, 0.3, 0.4, 0.5]),
                        lb=np.array([1e-4, 1e-4, 1e-5, 1e-4]),
                        ub=np.array([np.inf, np.inf, np.inf, np.inf]),
                        iprint=self.iprint).solve(self.solver)

    def jump_part_sim(self):
        """
        # simulation of jump part of the process
        # lambda is the jump intensity, eta is the jump size
        """
        up_jump = np.random.exponential(self.lam, size=(100, 100)) * self.eta
        down_jump = np.r_['-1', np.zeros((100, 1)), up_jump[:, :-1]] / self.eta**2
        
        return up_jump * down_jump
