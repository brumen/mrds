#   skew model for forward curves
#
import config
import pickle
import datetime as dt
import numpy as np
import scipy
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats 
import scipy.optimize 
import scipy.interpolate  # spline package
import openopt  # optimization solver
import multiprocessing as mp
import csv
import os
import calendar
import copy
# cuda
if config.CUDA_PRESENT: 
    import pycuda.curandom
    import pycuda.gpuarray as gpa
    import pycuda.cumath
    from pycuda.compiler import SourceModule
    # from cudanormal import cudanormal
    import curand

import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import Tkinter as tk

import vols
import vols_fast
import near_corr
import mrds_fast
import quartic_cy
import ds
if config.CUDA_PRESENT: 
    import cuda_ops
import opd_avx   # for speedups

# F skew implementation
if config.CUDA_PRESENT: 
    F_skew_el = open(config.work_dir + 'cuda/skew_tsf.c', 'r').read()
    F_skew_mod = SourceModule(F_skew_el)
    F_skew_fct = F_skew_mod.get_function('F_skew_tsf')


class MrdsError(Exception):
    """
    Base class for MRDS exceptions
    """
    pass


class InputError(MrdsError):
    """
    Exception raised for errors in the input
    Attributes:
    :param msg -- explanation of the error
    """
    def __init__(self, msg):
        self.msg = msg


class market_model():
    """
    abstrct market model class
    """
    pass

class mouse_press_state_machine():
    pass


def opt_fct_skew_wrap(arg, **kwarg):
    """
    # wrapper for the skew MRD model calibration function
    """
    return mrd_skew.opt_fct_skew(*arg, **kwarg)


class mrd_skew(market_model):
    """
    # nb_assets ... number of assets
    # tenors_list ... a _list_ of tenors of the forward contracts, each tenor curve is a numpy.array
    # forward_curves ... one forward curve for each tenors from the respective forward curve
    # atm_vols ... a list of atm vols for every forward curve
    # delta_vecs ... deltas for the vol surface (2nd argument in the vol_surface matrix), for each curve respectively
    # vol_surface ... self. expl.
    """

    # some helper functions
    def _list_len_fct_(self): # 
        return lambda e: len (e) 

    def _empty_list_fct(self, n):
        return [np.array([])] * n

    def _ones_list_fct(self, n, m):
        return [np.ones(m)] * n
    
    def _ones_matrix_fct (self, n, m, k):
        return [np.ones((m, k))] * n

    # n x m list of matrices of size k x l 
    def _list_list_matrix_fct (self, n, m, k, l, init_val):
        return [[init_val * np.ones((k, l))] * m] * n

    # same as a function above, but uses the factor numbers for every asset 
    def _list_list_factor_matrix (self, init_val):
        return [[init_val * np.ones((self.nb_factors_list[asset_1], self.nb_factors_list[asset_2]))
                 for asset_2 in range(self.nb_assets)]
                for asset_1 in range (self.nb_assets)]

    def __init__(self, nb_assets, date_today,
                 multi_thread_ind=False, model_skew_ln_ind='skew',
                 subset_idx=-1, solver_init=None,
                 max_iter=None, verbose='none', debug_mode=False,
                 black_vol_inverse_tol=1e-4, seed=None,
                 cuda_ind=False):

        if solver_init is None:
            self.solver = 'scipy_cobyla'  # initial solver
        else:
            self.solver = solver_init

        # maximum iterations for the NLP algorithm
        if max_iter is None:
            self.max_iter = 50  # default nb. of iterations
        else:
            self.max_iter = max_iter

        # verbosity, implies verbosity throughout the class operation
        # verbose in 'none', 'low', 'med', 'high'
        # 'none' - prints only the essentials
        # 'low' - prints some intermediary
        # 'med' - NOT used yet
        # 'high' - print also some of the intermediary results w/ optimization, etc.
        self.verbose = verbose
        if verbose == 'none':
            self.iprint = -1 
        else:
            self.iprint = 100

        # debug mode
        # prints more data if selected, default = False 
        self.debug_mode = debug_mode

        self.nb_factors_list = [2] * nb_assets  # factor placeholder for every asset
        self.valid_models = ['skew', 'ln_ln']
        self.model_skew_ln_ind = model_skew_ln_ind
        self.seed = seed
        self.subset_idx = subset_idx  # subset of indexes that one should take
        self.black_vol_inverse_tol = black_vol_inverse_tol  # tolerance when searching for black inverse vol
        self.nb_assets = nb_assets

        # dates
        self.date_today = date_today
        self.date_today_dt = ds.convert_str_datetime(self.date_today)

        # allocating empty lists
        self.forward_tenors_list = self._empty_list_fct(nb_assets)
        self.forward_tenors_dt_list = self._empty_list_fct(nb_assets)
        self.forward_tenors_code_list = self._empty_list_fct(nb_assets)
        self.option_tenors_list = self._empty_list_fct(nb_assets)
        self.option_tenors_dt_list = self._empty_list_fct(nb_assets)
        self.option_tenors_code_list = self._empty_list_fct(nb_assets)
        self.forward_curve_list = self._empty_list_fct(nb_assets)
        self.forward_curve_len = np.zeros(nb_assets, dtype=int) # length of forward curves

        self.atm_vol_list = self._empty_list_fct(nb_assets)
        self.cash_vol_list = self._empty_list_fct(nb_assets)
        self.vol_surface_list = self._empty_list_fct(nb_assets)
        self.vol_surface_name_list = self._empty_list_fct(nb_assets)
        self.vol_param_list = self._empty_list_fct(nb_assets)
        self.vol_obj_list = self._empty_list_fct(nb_assets)
        self.delta_vec_list = self._empty_list_fct(nb_assets)

        # mean reversion, vol. parameters
        self.sigma_vec_list = self._empty_list_fct(nb_assets)
        self.kappa_vec_list = self._empty_list_fct(nb_assets)
        self.sigma_lb_list = self._empty_list_fct(nb_assets)
        self.sigma_ub_list = self._empty_list_fct(nb_assets)
        self.kappa_lb_list = self._empty_list_fct(nb_assets)
        self.kappa_ub_list = self._empty_list_fct(nb_assets)
        self.beta_T_list = self._empty_list_fct(nb_assets)
        self.C_vec_list = self._empty_list_fct(nb_assets)
        # indicator functions - whether the values are updated
        # indicator function for the sigma, kappa calibration
        self.sigma_kappa_calib_indicator_list = np.repeat(False, nb_assets)
        self.skew_calib_indicator_list = np.repeat(False, nb_assets)
        self.sim_date_indicator = False
        self.days_nb_const_ind = False  # monthly dat numbers are constr.
        self.simulate_spot_rn_ind = False  # indic. for random number for spot sim.
        self.cash_corr = np.eye(self.nb_assets)

        self.fwd_curve_name = ['UNK'] * self.nb_assets
        self.vol_curve_name = ['UNK'] * self.nb_assets

        def factor_corr_mat_fct(self, asset_1, asset_2,
                                same_asset_corr=0.98,
                                diff_asset_corr=0.96):
            """
            initial corr. matrix
            """
            if asset_1 == asset_2:
                return vols.corr_hyp_sec_mat(same_asset_corr, range(self.nb_factors_list[asset_1]))
            else:
                return diff_asset_corr * np.ones((self.nb_factors_list[asset_1],
                                                  self.nb_factors_list[asset_2]))

        def factor_corr_mat_fct_lb_ub(self, asset_1, asset_2, lb_ub_ind='ub'):
            if lb_ub_ind is 'lb':
                lb_ub_fact = -0.999
            else:
                lb_ub_fact = 0.999

            if asset_1 == asset_2:
                tmp_1 = np.ones((self.nb_factors_list[asset_1], self.nb_factors_list[asset_1]))
                tmp_ut = np.triu(tmp_1, 1)
                tmp_lt = np.tril(tmp_1, -1)
                if lb_ub_ind is 'ub':
                    return tmp_1 - tmp_ut * 0.001 - tmp_lt * 0.001
                else:
                    return tmp_1 - tmp_ut * 1.999 - tmp_lt * 1.999
            else:
                return lb_ub_fact * np.ones((self.nb_factors_list[asset_1],
                                             self.nb_factors_list[asset_2]))
        # factor correlations matrices (also for cross factor correlations )
        self.factor_corr_mat_list = [[factor_corr_mat_fct(self, asset_1, asset_2)
                                      for asset_2 in range(nb_assets)]
                                     for asset_1 in range(nb_assets)]
        self.factor_corr_mat_lb_list = [[factor_corr_mat_fct_lb_ub(self, asset_1, asset_2, lb_ub_ind='lb')
                                         for asset_2 in range(nb_assets)]
                                        for asset_1 in range(nb_assets)]
        self.factor_corr_mat_ub_list = [[factor_corr_mat_fct_lb_ub(self, asset_1, asset_2, lb_ub_ind='ub')
                                         for asset_2 in range(nb_assets)]
                                        for asset_1 in range(nb_assets)]

        # market correlation list of lists
        self.market_corr_list = [self._empty_list_fct(self.nb_assets)] * self.nb_assets
        self.model_corr_list = [self._empty_list_fct (self.nb_assets)] * self.nb_assets
        # other parameters 
        self.multi_thread_ind = multi_thread_ind 
        self.simulation_times = None
        self.simulated_curves = self._empty_list_fct(self.nb_assets) # placeholder for all simulated curves
        self.simulated_curves_ind = [False] * self.nb_assets
        self.simulated_curves_nb = [None] * self.nb_assets
        self.forward_curve_ch = [None] * self.nb_assets
        self.sim_rv_ind = {'ind': False,
                           'nb_sim': 0}
        self.sim_rv = None
        self.cuda_ind = cuda_ind
        # if self.cuda_ind:
        #    # construct objects used for cuda generation
        #    self.rng_cuda = curand.create_gen_simple()
        # placeholder for simulated_curves_fom, indexed by tenors_list, asset_nb, sim_nb
        self.simulated_curves_fom_cuda = {}
        self.simulated_curves_fom_cpu = {}

    def map_coms_to_nbs(self, com_fwd_list):
        """
        maps com_fwd_list to numbers, i.e. [com
        :param com_fwd_list:
        :return self.coms_to_nbs: dict of coms to numbers {'WTI': 0, 'BRENT': 1, ...
        """
        self.coms_to_nbs = {com: com_nb for (com, com_nb)
                            in zip(com_fwd_list, range(len(com_fwd_list)))}
        self.nbs_to_coms = {com_nb: com for (com, com_nb)
                            in zip(com_fwd_list, range(len(com_fwd_list)))}

    def set_cash_vols(self, asset_nb, cash_vols):
        """
        sets the cash vols for the particular asset.
        req: len(cash_vols) == len(fwd_curve_list[asset_nb])
        """
        self.cash_vol_list[asset_nb] = cash_vols

    def update_sim_times(self, st_init):
        if type(st_init) == np.ndarray:
            self.simulation_times = st_init
            self.simulation_times_dt = [self.date_today_dt + dt.timedelta(int(np.round(stf * 365.)))
                                        for stf in self.simulation_times]
        elif (type(st_init) == list) and (type(st_init[0]) == dt.datetime):
            self.simulation_times = np.array([(st - self.date_today_dt).days/365.
                                              for st in st_init])
            self.simulation_times_dt = self.simulation_times
        elif (type(st_init) == list) and (type(st_init[0]) == str):
            self.simulation_times = np.array([(ds.convert_str_datetime(date_) -
                                              ds.convert_str_datetime(self.date_today)).days / 365.
                                              for date_ in st_init])
            self.simulation_times_dt = st_init
        elif (type(st_init) == list) and (type(st_init[0]) == float):
            self.simulation_times = np.array(st_init)
            self.simulation_times_dt = [self.date_today_dt + dt.timedelta(int(np.round(stf * 365.)))
                                        for stf in self.simulation_times]
        self.nb_time_steps = len(st_init)

    def update_fwd_curve(self, asset_nb, new_fwd):
        self.forward_curve_list[asset_nb] = new_fwd
        self.forward_curve_ch[asset_nb] = True

    def update_fwd_vol_curves(self, asset_nb=0, price_shock=1., vol_shock=1.):
        if (price_shock != 1.) or (vol_shock != 1.):
            self.forward_curve_list[asset_nb] *= price_shock
            self.atm_vol_list[asset_nb] *= vol_shock
            self.black_vol_calibration(asset_nb)
            if self.model_skew_ln_ind == 'skew':
                self.calibrate_skew_params(asset_nb)
            self.generate_large_corr_mat()

    def update_market_date(self, new_market_date, asset='all'):
        """
        change the model by changing the market date
        :param new_market_date: market date in string format
        """
        self.date_today = new_market_date
        self.date_today_dt = ds.convert_str_datetime(new_market_date)

        def update_one_asset(asset_ch):
            f_new = [(ft_dt, ft_idx) for (ft_dt, ft_idx)
                     in zip(self.option_tenors_dt_list[asset_ch],
                            range(self.forward_curve_len[asset_ch]))
                     if ft_dt > self.date_today_dt]
            f_idx = [ft_idx for (ft_dt, ft_idx) in f_new]

            # forward curves shortening
            self.forward_curve_list[asset_ch] = self.forward_curve_list[asset_ch][f_idx]
            self.forward_curve_len[asset_ch] = len(f_idx)
            self.forward_tenors_dt_list[asset_ch] = [self.forward_tenors_dt_list[asset_ch][i] for i in f_idx]
            self.forward_tenors_list[asset_ch] = [(ft_dt - self.date_today_dt).days / 365.25
                                                  for ft_dt in self.forward_tenors_dt_list[asset_ch]]
            self.forward_tenors_code_list[asset_ch] = [self.forward_tenors_code_list[asset_ch][i] for i in f_idx]
            # option/vol curves
            self.option_tenors_dt_list[asset_ch] = [self.option_tenors_dt_list[asset_ch][i] for i in f_idx]
            self.option_tenors_list[asset_ch] = np.array([(ft_dt - self.date_today_dt).days / 365.25
                                                 for ft_dt in self.option_tenors_dt_list[asset_ch]])
            self.option_tenors_code_list[asset_ch] = [self.option_tenors_code_list[asset_ch][i] for i in f_idx]
            # other parameters
            self.beta_T_list[asset_ch] = self.beta_T_list[asset_ch][f_idx]
            self.atm_vol_list[asset_ch] = self.atm_vol_list[asset_ch][f_idx]
            self.C_vec_list[asset_ch] = self.C_vec_list[asset_ch][f_idx]
            if self.cash_vol_list[asset_ch].shape != (0,):
                self.cash_vol_list[asset_ch] = self.cash_vol_list[asset_ch][f_idx]
            if self.vol_param_list[asset_ch].shape != (0,):
                self.vol_param_list[asset_ch] = self.vol_param_list[asset_ch][f_idx, :]
                self.vol_surface_list[asset_ch] = self.vol_surface_list[asset_ch][f_idx, :]
                self.vol_obj_list[asset_ch].extract_tenors(new_market_date, f_idx)

        if asset == 'all':
            f_idx_l = [update_one_asset(asset_ch) for asset_ch in range(self.nb_assets)]
            return f_idx_l
        else:
            f_idx = update_one_asset(asset)
            return f_idx

    def read_curve_vol_data_db(self, date_, asset_nb, fwd_curve, vol_curve,
                               sub_idx_rows=np.array([-1]),
                               adj_fwd_tenors_days=None,
                               adj_vol_tenors_days=None):

        self.fwd_curve_name[asset_nb] = fwd_curve
        self.vol_curve_name[asset_nb] = vol_curve
        self.vol_surface_name_list[asset_nb] = ds.vol_hash[vol_curve]
        fwd_vol_matched = ds.read_data_matched_tenors(date_, fwd_curve, vol_curve,
                                                      adj_fwd_tenors_days=adj_fwd_tenors_days,
                                                      adj_vol_tenors_days=adj_vol_tenors_days)

        self.forward_tenors_list[asset_nb] = fwd_vol_matched['fwd_tenors']
        self.forward_curve_list[asset_nb] = fwd_vol_matched['fwd_curve']
        self.forward_tenors_code_list[asset_nb] = fwd_vol_matched['fwd_tenors_code']
        self.forward_tenors_dt_list[asset_nb] = fwd_vol_matched['fwd_tenors_dt']
        self.forward_curve_len[asset_nb] = len(self.forward_curve_list[asset_nb])

        # vol curve data
        self.option_tenors_list[asset_nb] = fwd_vol_matched['option_tenors']
        self.option_tenors_code_list[asset_nb] = fwd_vol_matched['option_tenors_code']
        self.option_tenors_dt_list[asset_nb] = fwd_vol_matched['option_tenors_dt']

        if sub_idx_rows is not np.array([-1]):
            self.forward_tenors_list[asset_nb] = self.forward_tenors_list[asset_nb][sub_idx_rows]
            self.forward_curve_list[asset_nb] = self.forward_curve_list[asset_nb][sub_idx_rows]
            self.forward_tenors_code_list[asset_nb] = self.forward_tenors_code_list[asset_nb][sub_idx_rows]
            self.forward_tenors_dt_list[asset_nb] = self.forward_tenors_dt_list[asset_nb][sub_idx_rows]
            self.forward_curve_len[asset_nb] = len(sub_idx_rows)
            # vol curve data
            self.option_tenors_list[asset_nb] = self.option_tenors_list[asset_nb][sub_idx_rows]
            self.option_tenors_code_list[asset_nb] =  [self.option_tenors_code_list[asset_nb][elt]
                                                       for elt in sub_idx_rows]
            self.option_tenors_dt_list[asset_nb] = [self.option_tenors_dt_list[asset_nb][elt]
                                                    for elt in sub_idx_rows]

        if self.vol_surface_name_list[asset_nb] is 'JWSS7':
            self.atm_vol_list[asset_nb] = np.array(fwd_vol_matched['vol_surface_params'][sub_idx_rows, 0],
                                                   dtype=np.double)
            self.vol_param_list[asset_nb] = np.array(fwd_vol_matched['vol_surface_params'][sub_idx_rows, :],
                                                     dtype=np.double)
        elif self.vol_surface_name_list[asset_nb] is 'ATM':
            self.atm_vol_list[asset_nb] = np.array(fwd_vol_matched['vol_surface_params'][sub_idx_rows],
                                                   dtype=np.double)

    def one_zero_zero_mat(self, n, k):
        tmp = np.zeros((n,k))
        tmp[:, 0] = 1
        return tmp

    def set_other_params (self, asset_nb):
        """
        # this has to be run _AFTER_ the curves have been initialized
        """
        self.beta_T_list[asset_nb] = np.ones(self.forward_curve_len[asset_nb])
        self.forward_curve_corr = 1 # WRONG WRONG WRONG WHAT DOES THAT MEAN 
        # !!!! inital value of C is such that it produces nearly flat implied vol (1, 0,0)
        self.C_vec_list[asset_nb] = self.one_zero_zero_mat(self.forward_curve_len[asset_nb], 3)
        self.delta_vec_list[asset_nb] = np.arange(0.2, 0.9, 0.1)
        if self.vol_surface_name_list[asset_nb] == 'JWSS7':
            self.vol_obj_list[asset_nb] = vols.jw7_params(self.date_today,
                                                          self.fwd_curve_name[asset_nb],
                                                          self.vol_curve_name[asset_nb],
                                                          nb_fwds_taken=self.forward_curve_len[asset_nb])

            vo_curr = self.vol_obj_list[asset_nb]
            self.vol_surface_list[asset_nb] = vo_curr.implied_vol_all_fwd_standard(self.delta_vec_list[asset_nb])

    def implied_vol(self, asset_nb, fwd_nb, K_v, t):
        """
        computes implied vol for asset
        K_v ... vector of strikes
        t ... implied vol for that time
        """
        if self.vol_surface_name_list[asset_nb] == 'JWSS7':
            return self.vol_obj_list[asset_nb].implied_vol(fwd_nb, K_v, t)
        elif self.vol_surface_name_list[asset_nb] == 'ATM':
            return self.atm_vol_list[asset_nb][fwd_nb]
        elif self.vol_surface_name_list[asset_nb] == 'manual_vol':  # CHECK 0 HERE BELO
            sigma = self.atm_vol_list[asset_nb][0]
            delta_new = scipy.stats.norm.cdf(np.log(K_v/self.forward_curve_list[asset_nb][0])/(np.sqrt(t)*sigma))
            return self.vol_obj_list[asset_nb].implied_vol[0](delta_new)

    def implied_vol_atm(self, asset_nb, fwd_nb):
        """
        computes model implied vol for ATM for asset_nb and fwd_nb
        """
        if self.vol_surface_name_list[asset_nb] is 'JWSS7':
            return self.vol_obj_list[asset_nb].sigma_0
        elif self.vol_surface_name_list[asset_nb] is 'ATM':
            return self.atm_vol_list[asset_nb][fwd_nb]

    def read_discount_curve_db(self, date_):
        """
        obtains discount curve
        """
        res_ = ds.read_discount_curve(date_)
        self._discount_tenors = res_["disc_tenors_numeric"]
        self._discount_discount = res_["disc_curve"]
        self._discount_yield = res_['yield_rates']
        self._discount_function = res_['discount_function']

    def DF(self, t):
        """
        discount from self.date_today_dt to t
        t can be in a format:
          float
          string ... '20141114'
          datetime ... dt.datetime(...
        """
        if (type(t) is np.double) or (type(t) is float):
            time_diff = t
        elif type(t) is str:
            t_dt = ds.convert_str_datetime(t)
            time_diff = (t_dt - self.date_today_dt).days / 365.
        elif type(t) is dt.datetime:
            time_diff = (t - self.date_today_dt).days / 365.

        return scipy.interpolate.splev(time_diff, self._discount_function)

    def read_model_config_db(self, asset_nb,
                             nb_factors=2,
                             sigma_init=np.array([0.188, 0.101]),
                             sigma_lb=np.array([0.05, 0.01]),
                             sigma_ub=np.array([4., 1.]),
                             kappa_init=np.array([0.1, 0.5]),
                             kappa_lb=np.array([0.05, 0.01]),
                             kappa_ub=np.array([12., 1.]),
                             corr_init=np.array([[1., 0.5], [0.5, 1.]]),
                             corr_lb=np.array([[1., -0.99], [-0.99, 1.]]),
                             corr_ub=np.array([[1., 0.99], [0.99, 1.]]),
                             ):
        self.nb_factors_list[asset_nb] = nb_factors
        self.sigma_vec_list[asset_nb] = sigma_init
        self.kappa_vec_list[asset_nb] = kappa_init
        self.kappa_lb_list[asset_nb] = kappa_lb
        self.kappa_ub_list[asset_nb] = kappa_ub
        self.sigma_lb_list[asset_nb] = sigma_lb
        self.sigma_ub_list[asset_nb] = sigma_ub
        self.factor_corr_mat_list[asset_nb][asset_nb] = corr_init
        self.factor_corr_mat_lb_list[asset_nb][asset_nb] = corr_lb
        self.factor_corr_mat_ub_list[asset_nb][asset_nb] = corr_ub

    def overwrite_market_corr (self, asset_1, asset_2, overwr):
        """
        # overwrites read corr. with manual
        """
        #mark_len = len (self.market_corr_list[asset_1][asset_2])
        #if ( len(overwr) != mark_len ):
        #    print "Overwritten corr. vec. length does not match original corr. vec."
        #else:
        if self.debug_mode:
            print "Corr. vec overwritten with", overwr
        self.market_corr_list[asset_1][asset_2] = overwr

    def _construct_corr (self, mtx_size, theta_vector):
        """
        # constructs and upper triangular matrix from a vector theta_vector
        # first row is from the rho matrix
        """
        utm = np.triu(np.ones((mtx_size,mtx_size))) # upper triangular matrix
        utm_diag_ones = np.diag(np.diag(utm))
        utm = utm - utm_diag_ones
        utm[utm==1] = theta_vector
        utm = utm + utm.transpose() + utm_diag_ones
        return utm

    def _construct_corr_asset (self, asset_nb, theta_vector):
        return self._construct_corr(self.nb_factors_list[asset_nb], theta_vector)

    def _V_fct (self, asset_nb, kappa_vec, sigma_vec, corr_matrix, fwd_idx, t):
        """
        computes integrated square vol
        """
        sigma_vec_row = sigma_vec.reshape((1, self.nb_factors_list[asset_nb]))
        sigma_vec_col = sigma_vec.reshape((self.nb_factors_list[asset_nb], 1))
        kappa_vec_row = kappa_vec.reshape((1, self.nb_factors_list[asset_nb]))
        kappa_vec_col = kappa_vec.reshape((self.nb_factors_list[asset_nb], 1))

        sigma_m_1 = sigma_vec_col
        sigma_m_2 = sigma_vec_row
        kappa_m_1 = kappa_vec_row
        kappa_m_2 = kappa_vec_col
        cross_1 = self.beta_T_list[asset_nb][fwd_idx]**2 * sigma_m_2 * corr_matrix * sigma_m_1
        cross_2 = kappa_m_1 + kappa_m_2

        cross = cross_1 * (np.exp(-cross_2 * (self.forward_tenors_list[asset_nb][fwd_idx] - t)) -
                           np.exp(-cross_2 * self.forward_tenors_list[asset_nb][fwd_idx])) / cross_2
        return np.sum(cross)

    def V_fct_factor(self, asset_nb, factor_nb, fwd_idx, t_0, t_1):
        """
        computes integrated vol. V only for one factor (factor_nb)

        """
        kappa = self.kappa_vec_list[asset_nb][factor_nb]
        sigma = self.sigma_vec_list[asset_nb][factor_nb]
        beta = self.beta_T_list[asset_nb][fwd_idx]
        T = self.forward_tenors_list[asset_nb][fwd_idx]
        if kappa == 0.:
            return beta**2 * sigma**2 * (t_1 - t_0)
        else:
            return beta**2 * sigma**2 / (2. * kappa) * \
                   np.exp(-2. * kappa * T) * \
                   (np.exp(2. * kappa * t_1) - np.exp(2. * kappa * t_0))

    def _V_cross_factor(self, asset_nb, factor_1, factor_2, fwd_1, fwd_2, t_0, t_1):
        """
        computes cross integrated vol. V for only one factor
        t0, t1 ... vol is copmputed from t_0 to t_1
        """
        kappa_1 = self.kappa_vec_list[asset_nb][factor_1]
        kappa_2 = self.kappa_vec_list[asset_nb][factor_2]
        kappa_12 = kappa_1 + kappa_2
        sigma_1 = self.sigma_vec_list[asset_nb][factor_1]
        sigma_2 = self.sigma_vec_list[asset_nb][factor_2]
        rho_12 = self.factor_corr_mat_list[asset_nb][asset_nb][factor_1, factor_2]
        beta_1 = self.beta_T_list[asset_nb][fwd_1]
        beta_2 = self.beta_T_list[asset_nb][fwd_2]
        T_1 = self.forward_tenors_list[asset_nb][fwd_1]
        T_2 = self.forward_tenors_list[asset_nb][fwd_2]

        if kappa_12 == 0.:
            return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 * (t_1 - t_0)
        else:
            return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 / kappa_12 * \
                   (np.exp(-kappa_1 * (T_1 - t_1) - kappa_2 * (T_2-t_1)) -
                    np.exp(-kappa_1 * (T_1-t_0) - kappa_2 * (T_2-t_0)))

    def _V_total(self, asset_nb, fwd, t_0, t_1):
        v_total_1 = np.sum([self.V_fct_factor(asset_nb, factor_nb, fwd, t_0, t_1)
                            for factor_nb in range(self.nb_factors_list[asset_nb])])
        v_total_2 = np.sum(np.sum([[self._V_cross_factor(asset_nb,
                                                         factor_1, factor_2,
                                                         fwd, fwd, t_0, t_1)
                                    for factor_2 in range(factor_1 +1 , self.nb_factors_list[asset_nb])]
                                   for factor_1 in range(self.nb_factors_list[asset_nb])]))
        return v_total_1 + v_total_2

    def black_vol(self, asset_nb, kappa_vec, sigma_vec, corr_matrix, forward_mat_idx):
        return np.sqrt(self._V_fct(asset_nb, kappa_vec, sigma_vec, corr_matrix, forward_mat_idx,
                                   self.option_tenors_list[asset_nb][forward_mat_idx])  /
                       self.option_tenors_list[asset_nb][forward_mat_idx])

    # shows the black vol for the asset and future contract
    def black_vol_current(self, asset_nb, fwd_idx):
        """
        computes black vol until option maturity for the given parameters
        """
        return self.black_vol(asset_nb, self.kappa_vec_list[asset_nb],
                              self.sigma_vec_list[asset_nb],
                              self.factor_corr_mat_list[asset_nb][asset_nb],
                              fwd_idx)

    def V_fct_current(self, asset_nb, fwd_idx, t):
        return self._V_fct(asset_nb, self.kappa_vec_list[asset_nb],
                           self.sigma_vec_list[asset_nb],
                           self.factor_corr_mat_list[asset_nb][asset_nb],
                           fwd_idx, t)

    def black_vol_between(self, asset_nb, fwd, t_0, t_1):
        if type(t_0) is str:
            t_used_0 = ds.time_diff(self.date_today, t_0)
            t_used_1 = ds.time_diff(self.date_today, t_1)
        else:
            t_used_0 = t_0
            t_used_1 = t_1
        v_tot = self._V_total(asset_nb, fwd, t_used_0, t_used_1)
        return np.sqrt(v_tot/(t_used_1 - t_used_0))

    def black_vol_calibration(self, asset_nb):
        """
        calibrates kappa and sigma and rho parameters
        """
        model_black_vol = lambda kappa_vec, sigma_vec, rho_vec : \
            np.sum((np.array([self.black_vol(asset_nb, kappa_vec, sigma_vec,
                                             self._construct_corr_asset(asset_nb, rho_vec), T)
                              for T in range(self.forward_curve_len[asset_nb])])
                                 - self.atm_vol_list[asset_nb])**2)
        nbf = self.nb_factors_list[asset_nb]
        optim_fnc = lambda kappa_sigma_rho_vec: model_black_vol(kappa_sigma_rho_vec[0:nbf],
                                                                kappa_sigma_rho_vec[nbf:(2*nbf)],
                                                                kappa_sigma_rho_vec[(2*nbf):])

        # extracting the upper triangular part of the correlation matrix 
        fcm_init = self.factor_corr_mat_list[asset_nb][asset_nb]
        corr_init_ravel = np.triu(fcm_init, 1)[np.triu(fcm_init, 1) != 0]
        fcm_lb = self.factor_corr_mat_lb_list[asset_nb][asset_nb]
        corr_lb_ravel = np.triu(fcm_lb, 1)[np.triu(fcm_lb, 1) != 0]
        fcm_ub = self.factor_corr_mat_ub_list[asset_nb][asset_nb]
        corr_ub_ravel = np.triu(fcm_ub, 1)[np.triu(fcm_ub, 1) != 0]

        kappa_sigma_rho_init = np.concatenate([self.kappa_vec_list[asset_nb],
                                               self.sigma_vec_list[asset_nb],
                                               corr_init_ravel])
        kappa_sigma_rho_lb = np.concatenate([self.kappa_lb_list[asset_nb],
                                             self.sigma_lb_list[asset_nb],
                                             corr_lb_ravel])
        kappa_sigma_rho_ub = np.concatenate([self.kappa_ub_list[asset_nb],
                                             self.sigma_ub_list[asset_nb],
                                             corr_ub_ravel])

        # optimization run
        optim_pr = openopt.NLP(optim_fnc,
                               kappa_sigma_rho_init,
                               lb=kappa_sigma_rho_lb,
                               ub=kappa_sigma_rho_ub,
                               iprint=self.iprint)
        optim_res = optim_pr.solve(self.solver) 
        
        self.kappa_vec_list[asset_nb] = optim_res.xf[0:nbf]
        self.sigma_vec_list[asset_nb] = optim_res.xf[nbf:(2*nbf)]
        self.factor_corr_mat_list[asset_nb][asset_nb] = self._construct_corr_asset(asset_nb,
                                                                                   optim_res.xf[(2*nbf):])
        # set beta_T, indicator
        self.beta_T_list[asset_nb] = self.beta_T_calibration(asset_nb)
        self.sigma_kappa_calib_indicator_list[asset_nb] = True 
        self.check_black_vol_calib(asset_nb)  # check calibration
        return optim_res.xf

    def beta_T_calibration (self, asset_nb):
        """
        adjusts beta_T so that the atm vol is fitted perfectly
        (assuming that kappa, sigma, rho has already been calibrated)
        """
        return self.atm_vol_list[asset_nb] / \
               np.array(map(lambda forward_mat_idx: self.black_vol(asset_nb, self.kappa_vec_list[asset_nb],
                                                                   self.sigma_vec_list[asset_nb],
                                                                   self.factor_corr_mat_list[asset_nb][asset_nb],
                                                                   forward_mat_idx),
                            range(self.forward_curve_len[asset_nb]))).reshape(self.forward_curve_len[asset_nb])

    def check_black_vol_calib(self, asset_nb):
        """
        checks the black vol calibration
        """
        model_atm_vols = np.array([self.black_vol(asset_nb, self.kappa_vec_list[asset_nb],
                                                  self.sigma_vec_list[asset_nb],
                                                  self.factor_corr_mat_list[asset_nb][asset_nb], fwd)
                                   for fwd in range(self.forward_curve_len[asset_nb])])
        diff = scipy.linalg.norm(model_atm_vols - self.atm_vol_list[asset_nb])
        if diff > 1e-2:
            print "Calibration of ATM vols for asset nb. ", asset_nb, " FAILED. Diff=", diff
        else:
            print "Calibration of ATM vols for asset nb. ", asset_nb, " succeeded. Diff=", diff

    def __default_corr_mat__(self, asset_nb, exp_nb):
        """
        constructs the default correlation matrix
        the closer exp_nb is to 0, the more singular the matrix is
        and the correlation between forwards is closer to 1
        (does not need optimization)
        """
        nb_tenors = self.forward_curve_len[asset_nb]
        self.forward_curve_corr = [np.exp(-(np.abs(j-i)*exp_nb))
                                   for i in range(nb_tenors)
                                       for j in range(nb_tenors)]
        return self.forward_curve_corr

    def black_corr_within_curve (self, asset_nb, ind_1, ind_2):
        """
        the cummulative correlation between the ind_1-th and the ind_2-th future's contract
        up to the option time of the smallest of the two contracts
        asset_nb ... asset nb. for this curve
        """
        ind = np.min(ind_1, ind_2)
        opt_mat = self.option_tenors_list[asset_nb][ind] # opt_mat until the smallest one 
        corr = self.factor_corr_mat_list[asset_nb][asset_nb]
        kv = self.kappa_vec_list[asset_nb]
        sv = self.sigma_vec_list[asset_nb]
        ft = self.forward_tenors_list[asset_nb]
        a = np.array([corr[ind_1, ind_2] * sv[factor_nb_1] * sv[factor_nb_2] *
                      np.product(self.beta_T_list[asset_nb] ) *
                      (np.exp(- kv[factor_nb_1] * (ft[ind_1] - opt_mat) -
                              kv[factor_nb_2] * (ft[ind_2] - opt_mat) ) -
                       np.exp(- kv[factor_nb_1] * ft[ind_1] -
                              kv[factor_nb_2] * ft[ind_2])) /
                      (kv[factor_nb_1] + kv[factor_nb_2])
                      for factor_nb_1 in range(self.nb_factors_list[asset_nb])
                      for factor_nb_2 in range(self.nb_factors_list[asset_nb])])

        return (np.exp(np.sum(a)) - 1.) / self.black_vol(asset_nb, kv, sv, ind_1) / \
               self.black_vol(asset_nb, kv, sv, ind_2)  # covariance divided by 2 standard deviations

    def black_corr_within_curve_estimation(self, asset_nb, emp_corr_matrix):
        """
        this function performs the calibration
        """
        def model_corr(params):
            model_matrix = [self.black_corr_within_curve(asset_nb, ind_1, ind_2)
                            for ind_1 in range(self.forward_curve_len[asset_nb])
                            for ind_2 in range(self.forward_curve_len[asset_nb])]
            return np.sum(np.sum((model_matrix - emp_corr_matrix)** 2))
        initial_parameters = 1
        return scipy.optimize.brute(model_corr, initial_parameters)

    def black_corr_intra_curves(self, model_corr_mtx, curve_1, curve_2, tenor_1, tenor_2):
        """
        curve_1, curve_2 are different curves indices
        tenor_1 and tenor_2 are tenor indices
        """
        t_1 = self.option_tenors_list[curve_1][tenor_1]
        t_2 = self.option_tenors_list[curve_2][tenor_2]
        opt_mat = t_1 * (t_1 <= t_2) + t_2 * (t_2 < t_1)  # opt_mat until the smallest one
        kv1 = self.kappa_vec_list[curve_1]
        kv2 = self.kappa_vec_list[curve_2]
        sv1 = self.sigma_vec_list[curve_1]
        sv2 = self.sigma_vec_list[curve_2]
        ft1 = self.forward_tenors_list[curve_1]
        ft2 = self.forward_tenors_list[curve_2]

        a = np.array([model_corr_mtx[factor_nb_1, factor_nb_2] *
                    sv1[factor_nb_1] * sv2[factor_nb_2] *
                    self.beta_T_list[curve_1][tenor_1] * self.beta_T_list[curve_2][tenor_2] *
                    np.exp(- kv1[factor_nb_1] * ft1[tenor_1]
                           - kv2[factor_nb_2] * ft2[tenor_2]) *
                    (np.exp((kv1[factor_nb_1] + kv2[factor_nb_2]) * opt_mat) - 1) /
                    (kv1[factor_nb_1] + kv2[factor_nb_2] )
                    for factor_nb_1 in range(self.nb_factors_list[curve_1])
                    for factor_nb_2 in range(self.nb_factors_list[curve_2])])

        bv1 = np.sqrt(self.V_fct_current(curve_1, tenor_1, opt_mat))  # square of integrated variance
        bv2 = np.sqrt(self.V_fct_current(curve_2, tenor_2, opt_mat))
        return np.sum(a) / (bv1 * bv2)

    def black_corr_intra_curves_factors(self, model_corr_mtx, curve_1, curve_2,
                                        tenor_1, tenor_2,
                                        factor_nb_1, factor_nb_2, opt_mat):
        """
        same as function above (black_corr_intra_curves), but the factors are exposed
        curve_1, curve_2 are different curves indices
        tenor_1 and tenor_2 are tenor indices
        factor_nb_1, factor_nb_2 are factors for the two assets
        opt_mat ... until what maturity this is
        """
        kv1 = self.kappa_vec_list[curve_1]
        kv2 = self.kappa_vec_list[curve_2]
        sv1 = self.sigma_vec_list[curve_1]
        sv2 = self.sigma_vec_list[curve_2]
        ft1 = self.forward_tenors_list[curve_1]
        ft2 = self.forward_tenors_list[curve_2]
        tmp1 = model_corr_mtx[factor_nb_1, factor_nb_2] * \
               sv1[factor_nb_1] * sv2[factor_nb_2] * \
               self.beta_T_list[curve_1][tenor_1] * self.beta_T_list[curve_2][tenor_2] * \
               np.exp(- kv1[factor_nb_1] * ft1[tenor_1] - kv2[factor_nb_2] * ft2[tenor_2]) * \
               (np.exp((kv1[factor_nb_1] + kv2[factor_nb_2]) * opt_mat) - 1.) / \
               (kv1[factor_nb_1] + kv2[factor_nb_2])
        bv1 = np.sqrt(self.V_fct_factor(curve_1, factor_nb_1, tenor_1, 0., opt_mat))
        bv2 = np.sqrt(self.V_fct_factor(curve_2, factor_nb_2, tenor_2, 0., opt_mat))
        return tmp1 / (bv1 * bv2)

    def black_corr_intra_curves_calib(self, curve_1, curve_2):
        """
        calibrates the intra-curve correlations
        """
        black_corr_intra_curve_vector = lambda model_corr_mtx, curve_1, curve_2, corr_len: \
            np.array([self.black_corr_intra_curves(model_corr_mtx,
                                                   curve_1, curve_2,
                                                   tenor, tenor)
                      for tenor in xrange(corr_len)])
            
        black_corr_intra_curve_vector_optim = lambda model_corr_mtx, curve_1, curve_2, corr_len: \
            scipy.linalg.norm(black_corr_intra_curve_vector(model_corr_mtx, curve_1, curve_2, corr_len) -
                              self.market_corr_list[curve_1][curve_2])

        # !!!!! CHECK CHECK CHECK 
        # RELATIONSHIP BETWEEN size_1 and size_2 
        corr_len_real = len(self.market_corr_list[curve_1][curve_2])
        # !!!! WRONG WRONG - 2,2 SHOULD BE REMOVED IN MULTIPLE PLACES BELOW <- THIS HAS BEEN CORRECTED, CHECK IF IT WORKS
        curve_1_nb_fact = self.nb_factors_list[curve_1]  # this used to be 2
        curve_2_nb_fact = self.nb_factors_list[curve_2]  # this used to be 2
        optim_pr = openopt.NSP(lambda corr_mtx_ravel: black_corr_intra_curve_vector_optim(corr_mtx_ravel.reshape ((curve_1_nb_fact,curve_2_nb_fact)), \
                                                                                            curve_1, curve_2, corr_len_real), \
                                self.factor_corr_mat_list[curve_1][curve_2].ravel(), \
                                lb = self.factor_corr_mat_lb_list[curve_1][curve_2].ravel(), \
                                ub = self.factor_corr_mat_ub_list[curve_1][curve_2].ravel() )
        optim_res = optim_pr.solve(self.solver)  # solving done
        self.factor_corr_mat_list[curve_1][curve_2] = self.factor_corr_mat_list[curve_2][curve_1] = \
            np.array(optim_res.xf).reshape((curve_1_nb_fact, curve_2_nb_fact))  # assigning the matrix
        return np.array(optim_res.xf).reshape((2, 2))

    def __trunc_normal_above__(self, a):
        """
        computes the truncated E[ N^{0,1,2,3,4} * 1(N <a) ] where N std. normal
        in succession
        """
        if a < -1e10:
            return np.array([0., 0., 0., 0., 0.])
        elif a > 1e10:
            return np.array([1., 0., 1., 0., 3.])
        else:
            sqrt_2 = np.sqrt(2.)
            sqrt_2pi = np.sqrt(2. * np.pi)
            return np.array([scipy.stats.norm.cdf(a),
                             - np.exp(- a**2 / 2.0) / sqrt_2pi,
                             0.5 + 0.5 * scipy.special.erf(a / sqrt_2) -
                                 np.exp(-a**2/2.) * a / sqrt_2pi,
                             - (a**2 + 2.) * np.exp(- a**2/ 2.) / sqrt_2pi,
                             - a * (a**2 + 3.) * np.exp (- a**2 / 2.) / sqrt_2pi + \
                                 1.5 * (1. + scipy.special.erf(a / sqrt_2))])

    def __trunc_normal_below__(self, a):
        """
        computes the truncated E[ N^{0,1,2,3,4} * 1(N >a) ] where N std. normal
        in succession
        """
        return - self.__trunc_normal_above__(a) + np.array([1.0, 0.0, 1.0, 0.0, 3.0])

    def __trunc_normal_interval__(self, a, b):
        return self.__trunc_normal_above__(b) - self.__trunc_normal_above__(a)

    def deltas_to_strikes(self, asset_nb, tenor_nb):
        """
        converts deltas to strikes
        """
        integrated_vol = self.atm_vol_list[asset_nb][tenor_nb] * np.sqrt(self.option_tenors_list[asset_nb][tenor_nb])
        return np.exp((scipy.stats.norm.ppf(self.delta_vec_list[asset_nb]) - 0.5 * integrated_vol ) *
                      integrated_vol) * self.forward_curve_list[asset_nb][tenor_nb]

    def polynomial_european(self, asset_nb, C_vec, opt_mat_idx, strike, call_put_ind):
        """
        value of european call option in skew model with strike
        call_put_ind ... 1 for call, -1 for put
        """
        # obtaining the coefficients 
        A_V = self.skew_params(asset_nb, C_vec, opt_mat_idx)
        A = A_V[0:5]
        V = A_V[5]  # integrated volatility, not variance
        if call_put_ind == 1:  # call
            A0 = A[0] - strike 
            A1, A2, A3, A4 = A[1:5]
        else:  # put
            A0 = strike - A[0]
            A1, A2, A3, A4 = - A[1:5]

        if self.debug_mode:
            poly_roots = np.sort(np.poly1d([A4, A3, A2, A1, A0]).roots)
        else:
            if A4 == 0. and A3 == 0. and A2 == 0.:
                poly_roots = [-A0 / A1]
            elif A4 == 0. and A3 == 0.:
                poly_roots = np.sort(quartic_cy.QuadRoots(np.array([A2, A1, A0])))
            elif A4 == 0.:
                poly_roots = np.sort(quartic_cy.CubicRoots(np.array([A3, A2, A1, A0])))
            elif np.abs(A4) < 1e-6:
                poly_roots = np.sort(np.poly1d([A4, A3, A2, A1, A0]).roots)
            else:
                poly_roots = np.sort(quartic_cy.QuarticRoots(np.array([A4, A3, A2, A1, A0])))

        nb_real_roots = np.sum(poly_roots == poly_roots.real)  # number of real roots
        real_roots = poly_roots[poly_roots == poly_roots.real].real  # real roots only

        Asigma = np.array([A0, A1, A2, A3, A4]) * np.array([1., V, V**2, V**3, V**4])  # A multiplied by sigmas
        real_roots_tsf = real_roots / V  # equivalent of s in the document

        def integr_analy(self, real_roots_tsf, nb_real_roots, Asigma):
            # integrate the polynomial between the roots
            if nb_real_roots == 0:  # integrate polynomial function over whole of real axis
                if A4 > 0 or (A4 == 0 and A2 > 0) or (A4 == 0 and A2 == 0 and A0 > 0):
                    res = Asigma[0] + Asigma[2] + 3. * Asigma[4] 
                else: 
                    res = 0.0
            elif nb_real_roots == 1:
                if A3 > 0:
                    res = np.sum(self.__trunc_normal_below__(real_roots_tsf[0]) * Asigma)
                else:
                    res = np.sum(self.__trunc_normal_above__(real_roots_tsf[0]) * Asigma)
            elif (nb_real_roots == 2 or nb_real_roots == 3):  # integrate over 2 intervals
                if A4 > 0:
                    res = np.sum(self.__trunc_normal_above__(real_roots_tsf[0]) * Asigma) + \
                          np.sum(self.__trunc_normal_below__(real_roots_tsf[1]) * Asigma)
                if A4 < 0.:
                    res = np.sum(self.__trunc_normal_interval__(real_roots_tsf[0], real_roots_tsf[1]) * Asigma)
                if A4 == 0. and A3 != 0.:
                    if A3 > 0.:
                        res = np.sum(self.__trunc_normal_interval__(real_roots_tsf[0], real_roots_tsf[1]) * Asigma) + \
                            np.sum(self.__trunc_normal_below__(real_roots_tsf[2]) * Asigma)
                    else:  # A3 < 0
                        res = np.sum(self.__trunc_normal_above__(real_roots_tsf[0]) * Asigma) + \
                            np.sum(self.__trunc_normal_interval__(real_roots_tsf[1], real_roots_tsf[2]) * Asigma)
                if A4 == 0. and A3 == 0.:
                    if A2 < 0.:
                        res = np.sum(self.__trunc_normal_interval__(real_roots_tsf[0], real_roots_tsf[1]) * Asigma)
                    else:
                        res = np.sum(self.__trunc_normal_above__(real_roots_tsf[0]) * Asigma) + \
                            np.sum(self.__trunc_normal_below__(real_roots_tsf[1]) * Asigma)
            elif nb_real_roots == 4:  # integrate over 3 intervals
                if A4 > 0:
                    res = np.sum(self.__trunc_normal_above__(real_roots_tsf[0]) * Asigma ) + \
                          np.sum(self.__trunc_normal_below__(real_roots_tsf[3]) * Asigma ) + \
                          np.sum(self.__trunc_normal_interval__(real_roots_tsf[1], real_roots_tsf[2]) * Asigma)
                else: # A4 < 0
                    res = np.sum(self.__trunc_normal_interval__(real_roots_tsf[0], real_roots_tsf[1]) * Asigma) + \
                          np.sum(self.__trunc_normal_interval__(real_roots_tsf[2], real_roots_tsf[3]) * Asigma)
            return res

        def integr_num(A_V):
            # for testing purposes only, in prod. use the code above 
            A = A_V[0:5]
            V = A_V[5]  # integrated volatility, not variance
            if call_put_ind == 1:  # call
                A0 = A[0] - strike
                A1, A2, A3, A4 = A[1:5]
            else:  # put
                A0 = strike - A[0]
                A1, A2, A3, A4 = - A[1:5]

            return scipy.integrate.quad(lambda x: np.max([A0 + A1 * V * x + A2 * V**2 * x**2 +
                                                          A3 * V**3 * x**3 + A4 * V**4 * x**4, 0.]) / \
                                                  np.sqrt(2. * np.pi) * np.exp(- x**2 / 2.), -np.inf, np.inf)[0]

        disc_fact = self.DF(self.option_tenors_list[asset_nb][opt_mat_idx])
        if self.debug_mode:  # debug, selects the numeric approach
            return disc_fact * integr_num(A_V)
        else:  # prod. mode
            return disc_fact * integr_analy(self, real_roots_tsf, nb_real_roots, Asigma)

    def model_vol_surface(self, asset_nb, C_vec, fwd_idx):
        """
        computes model vols for asset_nb, C_vec, fwd_idx
        """
        strikes = self.deltas_to_strikes(asset_nb, fwd_idx)
        cp_ind = np.array([1 * (strike >= self.forward_curve_list[asset_nb][fwd_idx]) +
                           (-1) * (strike < self.forward_curve_list[asset_nb][fwd_idx])
                           for strike in strikes])
        price_vec_model = np.array([self.polynomial_european(asset_nb, C_vec,
                                                             fwd_idx, strike, cp)
                                    for strike, cp in zip(strikes, cp_ind)])
        bv_inv_fct = vols.black_vol_inverse
        bv = np.array([bv_inv_fct(self.forward_curve_list[asset_nb][fwd_idx],
                                  strike, opt_price,
                                  self.option_tenors_list[asset_nb][fwd_idx],
                                  self.DF(self.option_tenors_list[asset_nb][fwd_idx]),
                                  cp, self.black_vol_inverse_tol)
                       for opt_price, strike, cp in zip(price_vec_model, strikes, cp_ind)])
        return bv

    def disp_model_vols(self, asset, fwd):
        """
        plotting the model vols as you click the button
        """
        root = tk.Tk()  # main canvas
        # plot market vols as initial
        delta_x = self.deltas_to_strikes(asset, fwd)
        mv_y = self.vol_surface_list[asset][fwd]
        f = Figure(figsize=(5,4), dpi=100)
        a = f.add_subplot(111)
        line1, = a.plot(delta_x, mv_y)
        line2, = a.plot(delta_x, mv_y)  # 2 are needed
        # plot the graph
        dataPlot_canvas = FigureCanvasTkAgg(f, master=root)
        dataPlot_canvas.get_tk_widget().grid(row=0, column=0, rowspan=2)

        def refresh_model_vols(self, asset, fwd, c, a, canvas, li1, li2):
            mv1 = self.vol_surface_list[asset][fwd]
            bv = self.model_vol_surface(asset, c, fwd)  # model vols
            str1 = self.deltas_to_strikes(asset, fwd)
            # fastest way to draw data
            li1.set_xdata(str1)
            li1.set_ydata(bv)
            li2.set_xdata(str1)
            li2.set_ydata(mv1)
            canvas.draw()

        fct_update = lambda cc: refresh_model_vols(self, asset, fwd, [c1.get(), c2.get(), c3.get()],
                                                   a, dataPlot_canvas, line1, line2)

        c1 = tk.Scale(root, from_=-5., to=5., resolution=0.1, label='c0', command=fct_update)
        c2 = tk.Scale(root, from_=-25., to=25., resolution=0.2, label='c1', command=fct_update)
        c3 = tk.Scale(root, from_=-15., to=5., resolution=0.2, label='c2', command=fct_update)
        c1.grid(row=0, column=1)
        c2.grid(row=0, column=2)
        c3.grid(row=0, column=3)
        c1.set(self.C_vec_list[asset][fwd][0])
        c2.set(self.C_vec_list[asset][fwd][1])
        c3.set(self.C_vec_list[asset][fwd][2])
        dataPlot_canvas.show()
        root.mainloop()

    def disp_model_surf (self, asset, fwd):
        root = Tk() # main canvas
            
        c1 = Scale(root, from_=-2.0, to=2.0, resolution=0.1)
        c2 = Scale(root, from_=-5.0, to=5.0, resolution=0.2)
        c3 = Scale(root, from_=-5.0, to=5.0, resolution=0.2)
        
        c1.grid(row=0, column=1)
        c2.grid(row=0, column=2)
        c3.grid(row=0, column=3)
        
        # plot market vols as initial
        x = self.deltas_to_strikes(asset,fwd)
        y = self.vol_surface_list[asset][fwd]
        f = Figure(figsize=(5,4), dpi=100)
        a = f.add_subplot(111)
        a.plot (x,y) 
        
        
        # plot the graph 
        dataPlot_canvas = FigureCanvasTkAgg(f, master=root )
        dataPlot_canvas.show()
        dataPlot_canvas.get_tk_widget().grid(row=0,column=0, rowspan=2)
        
        
        # replot button 
        b1 = Button(root, text="replot", command=lambda : self.refresh_model_vols(asset, fwd, [c1.get(),c2.get(),c3.get()], a, \
                                                                                  dataPlot_canvas) ).grid(row=1,column=1, columnspan=3)
        
        root.mainloop()


    def opt_fct_skew(self, asset_nb, index_curr_tenor):
        """
        optimization function to minimize over the range 0: nb_tenors
        """
        def calib_obj_fct(self, asset_nb, C_vec, fwd_idx):
            imp_vol_vec_model = self.model_vol_surface(asset_nb, C_vec, fwd_idx)
            # penalize the calibrated funtion for values of C where positive forward prices.
            # penalization level is 10000
            #    imp_vol_vec_model - self.vol_surface_list[asset_nb][fwd_idx, :]
            return scipy.linalg.norm(imp_vol_vec_model -
                                     self.vol_surface_list[asset_nb][fwd_idx, :])  # vol surface given as rows
                                      
        skew_pr = openopt.NLP(lambda C: calib_obj_fct(self, asset_nb, C, index_curr_tenor),
                              self.C_vec_list[asset_nb][index_curr_tenor, :],
                              iprint=self.iprint)
        skew_sol = skew_pr.solve(self.solver)
        # checking the quality of calibration
        #    print "Model skew vols:", implied_skew_vol(self, asset_nb, skew_sol.xf, index_curr_tenor)
        #    print "Market skew vols:", self.vol_surface_list[asset_nb][index_curr_tenor,:]
        #    print "Difference(%): ", (implied_skew_vol(self, asset_nb, skew_sol.xf, index_curr_tenor)
        #                              - self.vol_surface_list[asset_nb][index_curr_tenor,:]) * 100.0
        #    print "Solution:", skew_sol.xf
        return skew_sol.xf 

    def calibrate_skew_params(self, asset_nb):
        """
        calibrates params C
        """
        if self.vol_surface_name_list[asset_nb] is 'ATM':  # return flat C = [1., 0., 0.]
            self.C_vec_list[asset_nb] = np.zeros((self.forward_curve_len[asset_nb], 3))
            self.C_vec_list[asset_nb][:, 0] = 1.
            return self.C_vec_list[asset_nb]
        else:
            if self.sigma_kappa_calib_indicator_list[asset_nb] == 0:
                self.black_vol_calibration(asset_nb)
            if not self.multi_thread_ind:
                C = [self.opt_fct_skew(asset_nb, T)
                     for T in range(self.forward_curve_len[asset_nb])]
            else:  # multithreading present
                nb_cores = mp.cpu_count()
                pool = mp.Pool(processes=nb_cores)
                curr_nb_tenors = len(self.forward_tenors_list[asset_nb])
                C = pool.map(opt_fct_skew_wrap,
                             zip([self] * curr_nb_tenors,
                                 [asset_nb] * curr_nb_tenors,
                                 range(curr_nb_tenors)))
                pool.close()
            self.C_vec_list[asset_nb] = np.array(C)  # C is a list, transformed now
        
        return C

    def generate_large_corr_mat(self, nb_steps=300):
        """
        generates the factor correlation matrix from a list of list of corr. matrices
        gathered in factor_corr_mat_list
        """
        cums = np.cumsum(self.nb_factors_list)  # borders between asset classes
        fact_sum = np.zeros(len(cums)+1)  # adding the first 0
        fact_sum[1:(len(cums)+1)] = cums
        fact_sum_last = fact_sum[-1]
        self.complete_corr_mat = np.zeros((fact_sum_last, fact_sum_last))
        # sets the large correlation building blocks
        for asset_1 in range(self.nb_assets):
            for asset_2 in range(self.nb_assets):
                self.complete_corr_mat[fact_sum[asset_1]:fact_sum[asset_1+1],
                                       fact_sum [asset_2]:fact_sum[asset_2+1]] = \
                    self.factor_corr_mat_list[asset_1][asset_2]

        # find the closest matrix that is positive semidefinite
        self.complete_corr_mat = near_corr.near_corr_simple(self.complete_corr_mat, nb_steps) # !!!! 30 STEPS IS THIS ENOUGH
        while not (np.linalg.eig(self.complete_corr_mat)[0] > 0.).all():
            d1, v1 = np.linalg.eig(self.complete_corr_mat)
            d1p = np.diag(np.maximum(d1, 1.e-16))
            self.complete_corr_mat = np.dot(v1, np.dot(d1p, v1.transpose()))
        # The correlation between simulated factors is not exactly the
        # correlation between brownian motions (which is complete_corr_mat above)
        # here we adjust the correlation 
        # self.large_sim_factors_corr_mat = zeros ((fact_sum_last, fact_sum_last))
        # for asset_1 in range (self.nb_assets):
        #    for asset_2 in range (self.nb_assets):
        #        for factor_1 in range ( ):
        #            for factor_2 in range ():
        #                tmp1 = (RRR )
        #                self.large_sim_factors_corr_mat[ fact_sum[asset_1] + factor_1, fact_sum[asset_2] + factor_2] = INIT HERE / 

    def find_1nb(self, asset_nb, t, dt_format=365.25):
        if type(t) is str:
            t_used = ds.time_diff(self.date_today, t, dt_format=dt_format)
        else:
            t_used = t
        t_smaller = [t for t in self.forward_tenors_list[asset_nb] if t < t_used]
        return len(t_smaller)

    def _var_covar_mtx(self, asset_nb, fwd_idx, i, j, t_idx, sim_times):
        """
        generate covar mtx, part of LN simulation, used in simulate_curves
        """
        if t_idx == 0:
            t_prev = 0.
        else:
            t_prev = sim_times[t_idx - 1]

        t_next = sim_times[t_idx]
        if i==j:
            return self.V_fct_factor(asset_nb, i, fwd_idx, t_prev, t_next)
        else:
            return self._V_cross_factor(asset_nb, i, j, fwd_idx, fwd_idx, t_prev, t_next)

    def simulate_curves(self, nb_simulations,
                        tenor_list=None,
                        set_seed=None, fom_ind=False,
                        cuda_ind=False):
        if cuda_ind:
            self.simulate_curves_cuda(nb_simulations,
                                      tenor_list=tenor_list,
                                      set_seed=set_seed)
        else:
            self.simulate_curves_cpu(nb_simulations,
                                     tenor_list=tenor_list,
                                     set_seed=set_seed,
                                     fom_ind=fom_ind)

    def simulate_curves_cpu(self, nb_simulations,
                            tenor_list=None,
                            set_seed=None, fom_ind=False):
        """
        simulate just one curve

        generates a 3-dimensional array
        0-th dimension: asset_nb
        1-st dimension: simulation times
        2-nd dimension: curve
        3-rd dimension: repeats of the curve

        fom_ind ... first of month indicator (used for tolling)
        """
        np.random.seed(set_seed)
        cums = np.cumsum(self.nb_factors_list)  # borders between asset classes
        fact_sum = np.zeros(len(cums)+1, dtype=np.int)  # adding the first 0
        fact_sum[1:(len(cums)+1)] = cums
        
        # preliminary work that can be done
        if tenor_list is None:
            fwd_c_len = [self.forward_curve_len[asset_nb] for asset_nb in range(self.nb_assets)]
        else:
            fwd_c_len = [len(tenor_for_asset) for tenor_for_asset in tenor_list]

        for asset_nb in range(self.nb_assets):  # looping over assets
            self.simulated_curves[asset_nb] = np.empty((len(self.simulation_times),
                                                        fwd_c_len[asset_nb], nb_simulations))
            if tenor_list is None:
                fwd_c_reshape = self.forward_curve_list[asset_nb].reshape(fwd_c_len[asset_nb], 1)
            else:
                fwd_c_reshape = self.forward_curve_list[asset_nb][tenor_list[asset_nb]].reshape(fwd_c_len[asset_nb], 1)

            if self.model_skew_ln_ind == 'skew':
                self.simulated_curves[asset_nb][0, :, :] = fwd_c_reshape
            else:  # ln-model
                # ln_model is simulated in logs, do the log now, convert the exp later
                self.simulated_curves[asset_nb][0, :, :] = np.log(fwd_c_reshape)

            self.forward_curve_ch[asset_nb] = False
            self.simulated_curves_nb[asset_nb] = nb_simulations
            self.simulated_curves_ind[asset_nb] = True

        X = [np.zeros((fwd_c_len[asset_nb], nb_simulations)) for asset_nb in range(self.nb_assets)]
        X_prev = [np.empty((fwd_c_len[asset_nb], nb_simulations)) for asset_nb in range(self.nb_assets)]

        # looping over time steps
        #   simulates ln process, basis for skew as well
        #   t_i ... idx of sim_time
        #   fact_sum ... factors of the individual assets
        for t_i in range(self.nb_time_steps):
            nb_factors = np.sum(self.nb_factors_list)
            simulated_rn = np.random.multivariate_normal(np.zeros(nb_factors),
                                                         self.complete_corr_mat,
                                                         size=nb_simulations)
            for asset_nb in range(self.nb_assets):
                nb_factors_asset = self.nb_factors_list[asset_nb]
                old_cov_mat = self.complete_corr_mat[fact_sum[asset_nb]:fact_sum[asset_nb+1],
                    fact_sum[asset_nb]:fact_sum[asset_nb+1]]
                old_chol = np.linalg.cholesky(old_cov_mat)
                sims_Z = simulated_rn[:, fact_sum[asset_nb]:fact_sum[asset_nb+1]].transpose()
                sims_Z_unit = np.dot(np.linalg.inv(old_chol), sims_Z)
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
                    delta_X = np.sum(np.dot(new_chol, sims_Z_unit), axis=0)

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

                    if self.model_skew_ln_ind is 'ln_ln':
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
                        # F_res = F_u * (1. + X_u + 0.5 * c1 * (X_u**2 - V_u) +
                        #                c2 * (X_u**3 - 3. * X_u * V_u) / 6. +
                        #                c3 * (X_u**4 - 6. * V_u * X_u**2 + 3. * V_u**2) / 24.)
                        # self.simulated_curves[asset_nb][t_i, tenor_idx, :] = F_res
                        # CHECK CHECK CHECK - THIS MIGHT BE WRONG
                        opd_avx.skew_fom(F_u, X_u, 0.5 * c1, V_u, c2/6., c3/24.,
                                         self.simulated_curves[asset_nb][t_i, tenor_idx, :],
                                         nb_simulations)

        if self.model_skew_ln_ind == 'ln_ln':
            for asset_nb in range(self.nb_assets):
                self.simulated_curves[asset_nb] = np.exp(self.simulated_curves[asset_nb])

    def simulate_curves_cuda(self, nb_simulations,
                             tenor_list=None,
                             set_seed=None):
        """
        simulate forward curves on cuda

        generates a 3-dimensional array
         0-th dimension: asset_nb
         1-st dimension: simulation times
         2-nd dimension: curve
         3-rd dimension: repeats of the curve
        """
        np.random.seed(set_seed)
        cums = np.cumsum(self.nb_factors_list)  # borders between asset classes
        fact_sum = np.zeros(len(cums)+1, dtype=np.int)  # adding the first 0
        fact_sum[1:(len(cums)+1)] = cums

        # preliminary work that can be done
        if tenor_list is None:
            fwd_c_len = [self.forward_curve_len[asset_nb] for asset_nb in range(self.nb_assets)]
        else:
            fwd_c_len = [len(tenor_for_asset) for tenor_for_asset in tenor_list]

        for asset_nb in range(self.nb_assets):  # looping over assets
            self.simulated_curves[asset_nb] = gpa.zeros((len(self.simulation_times),
                                                         fwd_c_len[asset_nb],
                                                         nb_simulations),
                                                        dtype=np.float32)

            if tenor_list is None:
                fwd_c = self.forward_curve_list[asset_nb].astype(np.float32)
            else:
                fwd_c = self.forward_curve_list[asset_nb][tenor_list[asset_nb]].astype(np.float32)

            if self.model_skew_ln_ind == 'skew':
                cuda_ops.vtpm_cols(fwd_c, self.simulated_curves[asset_nb][0, :, :], tm_ind='p')
                # for sim_idx in range(nb_simulations):
                #    self.simulated_curves[asset_nb][0, :, sim_idx] = fwd_c
            else:  # ln-model
                # ln_model is simulated in logs, do the log now, convert the exp later
                cuda_ops.vtpm_cols(np.log(fwd_c), self.simulated_curves[asset_nb][0, :, :], tm_ind='p')

            self.forward_curve_ch[asset_nb] = False
            self.simulated_curves_nb[asset_nb] = nb_simulations
            self.simulated_curves_ind[asset_nb] = True

        X = [gpa.zeros((fwd_c_len[asset_nb], nb_simulations), dtype=np.float32)
             for asset_nb in range(self.nb_assets)]
        X_prev = [gpa.empty((fwd_c_len[asset_nb], nb_simulations), dtype=np.float32)
                  for asset_nb in range(self.nb_assets)]

        nb_factors = np.sum(self.nb_factors_list)

        simulated_rn = gpa.empty((nb_factors, nb_simulations), dtype=np.float32)
        compl_corr_chol_gpu = gpa.to_gpu(np.linalg.cholesky(self.complete_corr_mat).astype(np.float32))
        g1 = curand.create_gen_simple()

        # looping over time steps
        #   simulates ln process, basis for skew as well
        #   t_i ... idx of sim_time
        #   fact_sum ... factors of the individual assets
        for t_i in range(self.nb_time_steps):
            # simulated_rn = np.random.multivariate_normal(np.zeros(nb_factors),
            #                                             self.complete_corr_mat,
            #                                             size=nb_simulations)
            # r1.fill_normal(simulated_rn_init)
            simulated_rn_init = gpa.empty((nb_factors, nb_simulations), dtype=np.float32)  # HAS TO BE HERE!!!
            curand.gen_eff_dev_rns(simulated_rn_init.size, np.longlong(simulated_rn_init.ptr), g1)
            cuda_ops.matmul(compl_corr_chol_gpu, simulated_rn_init, simulated_rn)

            for asset_nb in range(self.nb_assets):
                nb_factors_asset = self.nb_factors_list[asset_nb]
                old_cov_mat = self.complete_corr_mat[fact_sum[asset_nb]:fact_sum[asset_nb+1],
                                                     fact_sum[asset_nb]:fact_sum[asset_nb+1]]
                old_chol = np.linalg.cholesky(old_cov_mat)
                old_chol_inv = np.linalg.inv(old_chol)
                old_chol_inv_gpu = gpa.to_gpu(old_chol_inv.astype(np.float32))

                # extra matrices for using later
                sims_Z_unit = gpa.empty((fact_sum[asset_nb+1] - fact_sum[asset_nb], nb_simulations),
                                        dtype=np.float32)
                sims_Z_unit_mul = gpa.empty((fact_sum[asset_nb+1] - fact_sum[asset_nb], nb_simulations),
                                            dtype=np.float32)

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
                    new_chol_gpu = gpa.to_gpu(new_chol.astype(np.float32))

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
                        F_skew_fct(np.float32(F_u), np.float32(c1), np.float32(c2),
                                   np.float32(c3), np.float32(V_u), X_u,
                                   self.simulated_curves[asset_nb][t_i, tenor_idx, :],
                                   np.int32(nb_simulations),
                                   block=(1, 1, 1), grid=(nb_simulations, 1))
                        # F_res = F_u * (1. + X_u + 0.5 * c1 * (X_u**2 - V_u) +
                        #               c2 * (X_u**3 - 3. * X_u * V_u) / 6. +
                        #               c3 * (X_u**4 - 6. * V_u * X_u**2 + 3. * V_u**2) / 24.)
                        # self.simulated_curves[asset_nb][t_i, tenor_idx, :] = F_res.astype(np.float32)

        if self.model_skew_ln_ind == 'ln_ln':
            for asset_nb in range(self.nb_assets):
                self.simulated_curves[asset_nb] = pycuda.cumath.exp(self.simulated_curves[asset_nb])

    def simulate_curves_cuda_double(self, nb_simulations,
                                    tenor_list=None,
                                    set_seed=None):
        """
        simulate forward curves on cuda

        generates a 3-dimensional array
         0-th dimension: asset_nb
         1-st dimension: simulation times
         2-nd dimension: curve
         3-rd dimension: repeats of the curve
        """
        np.random.seed(set_seed)
        cums = np.cumsum(self.nb_factors_list)  # borders between asset classes
        fact_sum = np.zeros(len(cums)+1, dtype=np.int)  # adding the first 0
        fact_sum[1:(len(cums)+1)] = cums

        # preliminary work that can be done
        if tenor_list is None:
            fwd_c_len = [self.forward_curve_len[asset_nb] for asset_nb in range(self.nb_assets)]
        else:
            fwd_c_len = [len(tenor_for_asset) for tenor_for_asset in tenor_list]

        for asset_nb in range(self.nb_assets):  # looping over assets
            self.simulated_curves[asset_nb] = gpa.zeros((len(self.simulation_times),
                                                         fwd_c_len[asset_nb],
                                                         nb_simulations),
                                                        dtype=np.double)

            if tenor_list is None:
                fwd_c = self.forward_curve_list[asset_nb]
            else:
                fwd_c = self.forward_curve_list[asset_nb][tenor_list[asset_nb]]

            if self.model_skew_ln_ind == 'skew':
                cuda_ops.vtpm_cols(fwd_c, self.simulated_curves[asset_nb][0, :, :], tm_ind='p')
                # for sim_idx in range(nb_simulations):
                #    self.simulated_curves[asset_nb][0, :, sim_idx] = fwd_c
            else:  # ln-model
                # ln_model is simulated in logs, do the log now, convert the exp later
                cuda_ops.vtpm_cols(np.log(fwd_c), self.simulated_curves[asset_nb][0, :, :], tm_ind='p')

            self.forward_curve_ch[asset_nb] = False
            self.simulated_curves_nb[asset_nb] = nb_simulations
            self.simulated_curves_ind[asset_nb] = True

        X = [gpa.zeros((fwd_c_len[asset_nb], nb_simulations), dtype=np.double)
             for asset_nb in range(self.nb_assets)]
        X_prev = [gpa.empty((fwd_c_len[asset_nb], nb_simulations), dtype=np.double)
                  for asset_nb in range(self.nb_assets)]

        nb_factors = np.sum(self.nb_factors_list)
        simulated_rn = gpa.empty((nb_factors, nb_simulations), dtype=np.double)
        compl_corr_chol_gpu = gpa.to_gpu(np.linalg.cholesky(self.complete_corr_mat))
        g1 = curand.create_gen_simple()

        # looping over time steps
        #   simulates ln process, basis for skew as well
        #   t_i ... idx of sim_time
        #   fact_sum ... factors of the individual assets
        for t_i in range(self.nb_time_steps):
            # simulated_rn = np.random.multivariate_normal(np.zeros(nb_factors),
            #                                             self.complete_corr_mat,
            #                                             size=nb_simulations)
            # r1.fill_normal(simulated_rn_init)
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

    def simulate_1nb(self, nb_simulations, set_seed=None):
        """
        # simulate the 1NB (rolling) contract
        #
        # generates a 3-dimensional array
        # 0-th dimension: asset_nb
        # 1-st dimension: simulation times
        # 2-rd dimension: repeats of the curve
        """
        if self.simulation_times[-1] > self.forward_tenors_list[0][-1]:
            print "Last simulation time is larger than the largest forward tenor."
            self.simulated_curves = [] 
        else:
            self.simulate_curves(nb_simulations, set_seed)  # simulate curves
            curves = self.simulated_curves
            self.simulated_curves = self._empty_list_fct(self.nb_assets)  # removing the prev. sim. curves
            for asset_nb in range(self.nb_assets):
                self.simulated_curves[asset_nb] = np.empty((len(self.simulation_times), nb_simulations))
                for t_i in range(self.nb_time_steps):
                    current_nb = np.sum(self.forward_tenors_list[asset_nb] <= self.simulation_times[t_i])
                    self.simulated_curves[asset_nb][t_i, :] = curves[asset_nb][t_i, current_nb,:]

    def gen_days_number(self, asset_nb):
        """
        generates a dict of number of days per month for every tenor
        output:
           self.nb_days_month[m] for m being the m-th tenor in the model
        """
        if not self.days_nb_const_ind:  # if not yet constructed
            nb_fwds = len(self.forward_tenors_list[asset_nb])
            self.nb_days_month = {}
            beg_curr_month = self.date_today_dt - dt.timedelta(self.date_today_dt.day - 1)
            curr_month = beg_curr_month
            curr_month_y, curr_month_m = curr_month.year, curr_month.month
            for m in range(nb_fwds):  # m month
                self.nb_days_month[m] = calendar.monthrange(curr_month_y, curr_month_m)[1]
                if curr_month_m == 12:
                    curr_month_y += 1
                    curr_month_m = 1
                else:
                    curr_month_m += 1
            self.days_nb_const_ind = True  # indicator about the days number
        return self.nb_days_month

    def update_cash_corr(self, cash_corr_adj):
        self.cash_corr = cash_corr_adj

    def gen_spot_rn(self, nb_simulations, nb_days=65,
                    cuda_ind=False):
        if cuda_ind:
            self.gen_spot_rn_cuda(nb_simulations, nb_days=nb_days)
        else:
            self.gen_spot_rn_cpu(nb_simulations, nb_days=nb_days)

    def gen_spot_rn_cpu(self, nb_simulations, nb_days=65):
        """
        returns the random walk of nb_simulations and 31 days
        nb_days = 65  # suff. large
        """
        if not self.simulate_spot_rn_ind:  # random walk not yet initialized
            self.spot_rn = {}
            for day_idx in range(nb_days):
                self.spot_rn[day_idx] = np.random.multivariate_normal(np.zeros(self.nb_assets),
                                                                      self.cash_corr,
                                                                      size=nb_simulations)
            self.spot_rn_a = {}
            for asset_nb in range(self.nb_assets):
                self.spot_rn_a[asset_nb] = np.empty((nb_simulations, nb_days))
                for day_idx in range(nb_days):
                    self.spot_rn_a[asset_nb][:, day_idx] = self.spot_rn[day_idx][:, asset_nb]
            self.simulate_spot_rn_ind = True
        else:
            if self.spot_rn[0].shape[0] != nb_simulations:
                for day_idx in range(nb_days):
                    self.spot_rn[day_idx] = np.random.multivariate_normal(np.zeros(self.nb_assets),
                                                                          self.cash_corr,
                                                                          size=nb_simulations)
                    self.spot_rn_a = {}
                    for asset_nb in range(self.nb_assets):
                        self.spot_rn_a[asset_nb] = np.empty((nb_simulations, nb_days))
                        for day_idx in range(nb_days):
                            self.spot_rn_a[asset_nb][:, day_idx] = self.spot_rn[day_idx][:, asset_nb]

    def gen_spot_rn_cuda(self, nb_simulations, nb_days=65):
        """
        returns the random walk of nb_simulations and 31 days on cuda
        nb_days = 65  - something suff. large for a month (2 blocks per day)
        """
        g1 = curand.create_gen_simple()  # pycuda.curandom.XORWOWRandomNumberGenerator()
        if not self.simulate_spot_rn_ind:  # random walk not yet initialized
            self.spot_rn = {}
            for day_idx in range(nb_days):
                #self.spot_rn[day_idx] = np.random.multivariate_normal(np.zeros(self.nb_assets),
                #                                                      self.cash_corr,
                #                                                      size=nb_simulations)
                spot_rn_init = gpa.empty((self.nb_assets, nb_simulations), dtype=np.float32)
                cash_corr_gpu = gpa.to_gpu(np.linalg.cholesky(self.cash_corr).astype(np.float32))
                self.spot_rn[day_idx] = gpa.empty((nb_simulations, self.nb_assets), dtype=np.float32)
                # r1.fill_normal(spot_rn_init)  # fill spot_rn_init w/ random numbers
                curand.gen_eff_dev_rns(spot_rn_init.size, np.longlong(spot_rn_init.ptr), g1)
                cuda_ops.matmul(cash_corr_gpu, spot_rn_init, self.spot_rn[day_idx])

            self.spot_rn_a = {}
            for asset_nb in range(self.nb_assets):
                self.spot_rn_a[asset_nb] = gpa.empty((nb_simulations, nb_days), dtype=np.float32)
                for day_idx in range(nb_days):
                    self.spot_rn_a[asset_nb][:, day_idx] = self.spot_rn[day_idx][:, asset_nb]
            
            self.simulate_spot_rn_ind = True
        else:
            if self.spot_rn[0].shape[0] != nb_simulations:
                for day_idx in range(nb_days):
                    #self.spot_rn[day_idx] = np.random.multivariate_normal(np.zeros(self.nb_assets),
                    #                                                      self.cash_corr,
                    #                                                      size=nb_simulations)
                    spot_rn_init = gpa.empty((self.nb_assets, nb_simulations), dtype=np.float32)
                    cash_corr_gpu = gpa.to_gpu(np.linalg.cholesky(self.cash_corr).astype(np.float32))
                    self.spot_rn[day_idx] = gpa.empty((self.nb_assets, nb_simulations), dtype=np.float32)
                    # r1.fill_normal(spot_rn_init)
                    curand.gen_eff_dev_rns(spot_rn_init.size, np.longlong(spot_rn_init.ptr), g1)
                    cuda_ops.matmul(cash_corr_gpu, spot_rn_init, self.spot_rn[day_idx])
                    self.spot_rn_a = {}
                    for asset_nb in range(self.nb_assets):
                        self.spot_rn_a[asset_nb] = gpa.empty((nb_simulations, nb_days), dtype=np.float32)
                        for day_idx in range(nb_days):
                            self.spot_rn_a[asset_nb][:, day_idx] = self.spot_rn[day_idx][:, asset_nb]

    def simulate_spot(self, asset_nb, nb_simulations, sest_seed=None):
        """
        simulate daily spot using for all tenors the cash_vols for asset asset_nb
        """
        if self.simulation_times is not None:
            saving_done = True
            save_sim_times = copy.deepcopy(self.simulation_times)
        else:
            saving_done = False

        self.update_sim_times(self.option_tenors_list[asset_nb])
        self.gen_days_number(asset_nb)
        self.gen_spot_rn(nb_simulations)
        spot_sims = {}
        self.simulate_curves(nb_simulations)
        all_sims_asset = self.simulated_curves[asset_nb]  # forwards until the end of the month
        for fwd_tenor_nb, cash_vol_tenor in enumerate(self.cash_vol_list[asset_nb]):
            nb_days_m = self.nb_days_month[fwd_tenor_nb]
            if fwd_tenor_nb != (self.forward_curve_len[asset_nb] - 1):
                # not at the end of the month
                month_start = self.forward_tenors_list[asset_nb][fwd_tenor_nb]
                month_end = self.forward_tenors_list[asset_nb][fwd_tenor_nb+1]
            else:
                month_start = self.forward_tenors_list[asset_nb][fwd_tenor_nb]
                month_end = month_start + 1./12.

            days = np.linspace(month_start, month_end, nb_days_m)
            day_len = days[1] - days[0]
            fom_sims = all_sims_asset[fwd_tenor_nb, fwd_tenor_nb, :]
            W_days = np.cumsum(np.sqrt(day_len) * self.spot_rn[:, :nb_days_m], axis=1)
            spot_sims[fwd_tenor_nb] = np.transpose(fom_sims.reshape((len(fom_sims), 1)) *
                                                   np.exp(-0.5 * cash_vol_tenor**2 * days + cash_vol_tenor * W_days))

        if saving_done:
            self.simulation_times = save_sim_times
        else:
            self.simulation_times = None
        return spot_sims

    def simulate_curves_fom(self, asset_nb, nb_simulations,
                            tenors_list=None,
                            set_seed=None,
                            cuda_ind=False):
        if cuda_ind:
            fom_sims = self.simulate_curves_fom_cuda(asset_nb, nb_simulations,
                                                     tenors_list=tenors_list,
                                                     set_seed=set_seed)

            # self.simulated_curves_fom_cuda[(asset_nb, tenors_list, nb_simulations)] = fom_sims
            return fom_sims
        else:
            fom_sims = self.simulate_curves_fom_cpu(asset_nb, nb_simulations,
                                                    tenors_list=tenors_list,
                                                    set_seed=set_seed)
            # self.simulated_curves_fom_cpu[(asset_nb, tenors_list, nb_simulations)] = fom_sims
            return fom_sims

    def simulate_curves_fom_cpu(self, asset_nb, nb_simulations,
                                tenors_list=None,
                                set_seed=None):
        """
        simulate first of month curves
        generates a list of 2 dim arrays:
           1-st dim: tenor
           2-nd dim: simulation
        tenors_list: list of tenors for asset asset_nb that should be simulated
        """
        np.random.seed(set_seed)
        cums = np.cumsum(self.nb_factors_list)
        fact_sum = np.zeros(len(cums)+1, dtype=np.int)
        fact_sum[1:(len(cums)+1)] = cums

        if tenors_list is None:
            fwd_c_len = self.forward_curve_len[asset_nb]
            sim_times = self.option_tenors_list[asset_nb]
            tenors_used = range(fwd_c_len)
        else:
            fwd_c_len = len(tenors_list)
            sim_times = self.option_tenors_list[asset_nb][tenors_list]
            tenors_used = tenors_list

        sim_fom = np.empty((fwd_c_len, nb_simulations))

        def _var_covar_mtx_simple(self, asset_nb, fwd_idx, i, j, t_idx, sim_times):
            """
            generates covar mtx from 0 to sim_times[t_idx], part of LN simulation
            """
            t_prev = 0.
            t_next = sim_times[t_idx]
            if i==j:
                return self.V_fct_factor(asset_nb, i, fwd_idx, t_prev, t_next)
            else:
                return self._V_cross_factor(asset_nb, i, j, fwd_idx, fwd_idx, t_prev, t_next)

        # looping over tenors
        #    t_i ... idx of sim_time (also tenor)
        #    fact_sum ... factors of the individual assets
        for t_i, t_nb in enumerate(tenors_used):
            tenor_nb = t_nb
            t_curr = sim_times[t_i]
            F_curr = self.forward_curve_list[asset_nb][tenor_nb]
            nb_factors = np.sum(self.nb_factors_list)  # total nb. of factors
            simulated_rn = np.random.multivariate_normal(np.zeros(nb_factors), self.complete_corr_mat,
                                                         size=nb_simulations)
            nb_factors_asset = self.nb_factors_list[asset_nb]
            new_cov_mat = np.array([[_var_covar_mtx_simple(self, asset_nb, tenor_nb, i, j, t_i, sim_times)
                                    for j in range(nb_factors_asset)]
                                    for i in range(nb_factors_asset)])

            new_chol = np.linalg.cholesky(new_cov_mat)
            old_cov_mat = self.complete_corr_mat[fact_sum[asset_nb]:fact_sum[asset_nb+1],
                                                 fact_sum[asset_nb]:fact_sum[asset_nb+1]]
            old_chol = np.linalg.cholesky(old_cov_mat)

            sims_Z = simulated_rn[:, fact_sum[asset_nb]:fact_sum[asset_nb+1]].transpose()
            sims_Z_unit = np.dot(np.linalg.inv(old_chol), sims_Z)
            delta_X = np.sum(np.dot(new_chol, sims_Z_unit), axis=0)
            qv = np.sum([[self._V_cross_factor(asset_nb, factor_1, factor_2, tenor_nb, tenor_nb, 0., t_curr)
                          for factor_1 in range(nb_factors_asset)]
                         for factor_2 in range(nb_factors_asset)])

            if self.model_skew_ln_ind is 'ln_ln':
                sim_fom[t_i, :] = F_curr * np.exp(delta_X - 0.5 * qv)
            else:
                c1 = self.C_vec_list[asset_nb][tenor_nb, 0]
                c2 = self.C_vec_list[asset_nb][tenor_nb, 1]
                c3 = self.C_vec_list[asset_nb][tenor_nb, 2]
                # sim_fom[t_i, :] = F_curr * \
                #                    (1. + delta_X + 0.5 * c1 * (delta_X**2 - qv) +
                #                    c2 * (delta_X**3 - 3. * delta_X * qv) / 6. +
                #                    c3 * (delta_X**4 - 6. * qv * delta_X**2 + 3. * qv**2) / 24.)
                opd_avx.skew_fom(F_curr, delta_X, 0.5 * c1, qv, c2/6., c3/24.,
                                 sim_fom[t_i, :], nb_simulations)

        return sim_fom

    def simulate_curves_fom_cuda(self, asset_nb, nb_simulations,
                                 tenors_list=None,
                                 set_seed=None):
        """
        simulate first of month curves
        generates a list of 3 dim arrays:
           1-st dim: tenor
           2-nd dim: simulation
        """
        np.random.seed(set_seed)
        cums = np.cumsum(self.nb_factors_list)
        fact_sum = np.zeros(len(cums)+1, dtype=np.int)
        fact_sum[1:(len(cums)+1)] = cums
        sim_times = self.option_tenors_list[asset_nb]
        sim_fom = gpa.empty((self.forward_curve_len[asset_nb], nb_simulations), dtype=np.float32)

        def _var_covar_mtx_simple(self, asset_nb, fwd_idx, i, j, t_idx, sim_times):
            """
            generates covar mtx from 0 to sim_times[t_idx], part of LN simulation
            """
            t_prev = 0.
            t_next = sim_times[t_idx]
            if i == j:
                return self.V_fct_factor(asset_nb, i, fwd_idx, t_prev, t_next)
            else:
                return self._V_cross_factor(asset_nb, i, j, fwd_idx, fwd_idx, t_prev, t_next)

        g1 = curand.create_gen_simple()
        # looping over tenors
        #    t_i ... idx of sim_time (also tenor)
        #    fact_sum ... factors of the individual assets
        for t_i, t_curr in enumerate(sim_times):
            tenor_nb = t_i
            F_curr = self.forward_curve_list[asset_nb][tenor_nb].astype(np.float32)
            nb_factors = np.sum(self.nb_factors_list)  # total nb. of factors
            # simulated_rn = np.random.multivariate_normal(np.zeros(nb_factors), self.complete_corr_mat,
            # these lines replace one line above                                             size=nb_simulations)
            simulated_rn_init = gpa.empty((nb_factors, nb_simulations), dtype=np.float32)
            simulated_rn = gpa.empty((nb_factors, nb_simulations), dtype=np.float32)
            curand.gen_eff_dev_rns(simulated_rn_init.size, np.longlong(simulated_rn_init.ptr), g1)
            compl_corr_chol_gpu = gpa.to_gpu(np.linalg.cholesky(self.complete_corr_mat).astype(np.float32))
            cuda_ops.matmul(compl_corr_chol_gpu, simulated_rn_init, simulated_rn)
            # simulated_rn = cuda_ops.bdcast(compl_corr_chol_gpu, simulated_rn_init).transpose()
            simulated_rn = simulated_rn.transpose()  # this is done very fast 
            nb_factors_asset = self.nb_factors_list[asset_nb]
            new_cov_mat = np.array([[_var_covar_mtx_simple(self, asset_nb, tenor_nb, i, j, t_i, sim_times)
                                    for j in range(nb_factors_asset)]
                                        for i in range(nb_factors_asset)])
            new_chol = np.linalg.cholesky(new_cov_mat)
            new_chol_gpu = gpa.to_gpu(new_chol).astype(np.float32)
            old_cov_mat = self.complete_corr_mat[fact_sum[asset_nb]:fact_sum[asset_nb+1],
                                                 fact_sum[asset_nb]:fact_sum[asset_nb+1]]
            old_chol = np.linalg.cholesky(old_cov_mat)
            sims_Z = simulated_rn[:, fact_sum[asset_nb]:fact_sum[asset_nb+1]].transpose()
            sims_Z_unit = gpa.empty_like(sims_Z)
            matmul_used = gpa.to_gpu(np.linalg.inv(old_chol)).astype(np.float32)
            # sims_Z_unit = np.dot(np.linalg.inv(old_chol), sims_Z)
            cuda_ops.matmul(matmul_used, sims_Z, sims_Z_unit)

            # TO IMPROVE< THISI IS SLOW
            # delta_X = gpa.to_gpu(np.sum(np.dot(new_chol, sims_Z_unit), axis=0).astype(np.float32))
            delta_X_pre = gpa.empty_like(sims_Z_unit)
            cuda_ops.matmul(new_chol_gpu, sims_Z_unit, delta_X_pre)
            delta_X = cuda_ops.colsum_cuda_last(delta_X_pre)
            qv = np.sum([[self._V_cross_factor(asset_nb, factor_1, factor_2, tenor_nb, tenor_nb, 0., t_curr)
                          for factor_1 in range(nb_factors_asset)]
                         for factor_2 in range(nb_factors_asset)]).astype(np.float32)

            if self.model_skew_ln_ind is 'ln_ln':
                sim_fom[t_i, :] = F_curr * np.exp(delta_X - 0.5 * qv)
            else:
                c1 = self.C_vec_list[asset_nb][tenor_nb, 0].astype(np.float32)
                c2 = self.C_vec_list[asset_nb][tenor_nb, 1].astype(np.float32)
                c3 = self.C_vec_list[asset_nb][tenor_nb, 2].astype(np.float32)
                # s1 = np.float32(3 * qv**2)
                # new_sim = F_curr * \
                #    (1. + delta_X + c1 * (delta_X**2 - qv) / 2. +
                #     c2 * (delta_X**3 - delta_X * 3*qv) / 6. +
                #     c3 * (delta_X**4 - delta_X**2 * 6*qv + s1) / 24.)
                # sim_fom[t_i, :] = new_sim
                F_skew_fct(F_curr, c1, c2, c3, qv, delta_X, sim_fom[t_i, :],
                           np.int32(nb_simulations),
                           block=(1, 1, 1), grid=(nb_simulations, 1))
        return sim_fom

    def generate_days_vecs(self, hours_partition, days_partition, cuda_ind=False):
        """
        generate days for simulate_spot_blocks
        """
        # construct the equiv. of days = range(31)/365.25
        days = np.array([0.])
        for day in range(31):  # all possible days
            day_week = np.mod(day, 7)
            hours_for_day_week = [hp for (hp, dp) in zip(hours_partition, days_partition)
                                  if day_week in dp][0]
            days = np.append(days, days[-1] + np.cumsum(hours_for_day_week)/24./365.25)
        days_d = gpa.to_gpu(days).astype(np.float32)
        if cuda_ind:
            days_diff = gpa.empty(len(days), dtype=np.float32)
            days_diff[0] = np.array(0., dtype=np.float32)
            days_diff[1:] = np.diff(days).astype(np.float32)
        else:
            days_diff = np.empty(len(days))
            days_diff[0] = 0.
            days_diff[1:] = np.diff(days)

        days_diff_l = len(days_diff)

        return days, days_d, days_diff, days_diff_l

    def simulate_spot_blocks_all(self, nb_simulations,
                                 days_partition, hours_partition,
                                 tenors_chosen=None,
                                 set_seed=None,
                                 cuda_ind=False):
        """
        same as simulate_spot_blocks, but for all blocks
        """
        days_tuple = self.generate_days_vecs(hours_partition,
                                             days_partition,
                                             cuda_ind=cuda_ind)

        return [self.simulate_spot_blocks(asset_nb, nb_simulations,
                                          days_partition, hours_partition,
                                          days_tuple,
                                          tenors_chosen=tenors_chosen,
                                          set_seed=set_seed,
                                          cuda_ind=cuda_ind)
                for asset_nb in range(self.nb_assets)]

    def simulate_spot_blocks(self, asset_nb, nb_simulations,
                             days_partition, hours_partition,
                             days_tuple,
                             tenors_chosen=None,
                             set_seed=None,
                             cuda_ind=False):
        """
        used for a tolling model.
        days_partition ... [[0, 1, 2, 3, 4], [5, 6]]
        hours_partition ... [hours for blocks [[6, 18], [12, 12]]
        """
        # construct the equiv. of days = range(31)/365.25
        days, days_d, days_diff, days_diff_l = days_tuple
        fom_sims_all = self.simulate_curves_fom(asset_nb, nb_simulations,
                                                tenors_list=tenors_chosen,
                                                set_seed=set_seed,
                                                cuda_ind=cuda_ind)
        self.gen_days_number(asset_nb)
        self.gen_spot_rn(nb_simulations, cuda_ind=cuda_ind)

        spot_sims = {}
        if tenors_chosen is None:
            cv_tenors = zip(range(len(self.cash_vol_list[asset_nb])), self.cash_vol_list[asset_nb])
        else:
            cv_tenors = zip(tenors_chosen, self.cash_vol_list[asset_nb][tenors_chosen])

        if cuda_ind:  # cuda usage
            w_days = pycuda.cumath.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l]
            cuda_ops.cumsum_cuda(w_days)
            for fwd_tenor_nb, cash_vol_tenor in cv_tenors:
                # fom in column format
                fom_sims = fom_sims_all[fwd_tenor_nb, :]   # row vec
                mult_1 = np.float32(-0.5 * cash_vol_tenor**2)
                mult_2 = np.float32(cash_vol_tenor)
                col_vec = pycuda.cumath.exp(days_d * mult_1 + w_days * mult_2)
                # transpose is used
                spot_sims[fwd_tenor_nb] = cuda_ops.vtpv(fom_sims, col_vec, tm_ind='t',
                                                        transpose_ind=True).transpose()
        else:  # no cuda
            w_days = np.cumsum(np.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l],
                               axis=1)
            for fwd_tenor_nb, cash_vol_tenor in cv_tenors:
                # fom in column format
                fom_sims = fom_sims_all[fwd_tenor_nb, :].reshape((len(fom_sims_all[fwd_tenor_nb, :]), 1))
                spot_sims[fwd_tenor_nb] = np.transpose(fom_sims *
                                                       np.exp(-0.5 * cash_vol_tenor**2 * days +
                                                              cash_vol_tenor * w_days))

        return spot_sims

    def simulate_spot_blocks_from_fom(self, fom_sims_all, asset_nb, m, nb_simulations,
                                      days_partition, hours_partition,
                                      days_tuple,
                                      tenors_chosen=None,
                                      set_seed=None,
                                      cuda_ind=False):
        """
        used for a tolling model, generates spot blocks of month m from fom_sims_all
        :param m: month to simulate spot block from
        :param days_partition: partition of a week, i.e. [[0, 1, 2, 3, 4], [5, 6]]
        :param hours_partition: hours for blocks [[6, 18], [12, 12]]
        :param days_tuple: tuple of days, days_d???, days_diff, days_diff_l
        """
        # construct the equiv. of days = range(31)/365.25
        days, days_d, days_diff, days_diff_l = days_tuple
        self.gen_days_number(asset_nb)
        self.gen_spot_rn(nb_simulations, cuda_ind=cuda_ind)

        cv_m = self.cash_vol_list[asset_nb][m]  # cash vol for month m
        fom_sims_used = fom_sims_all[asset_nb]

        if cuda_ind:  # cuda usage
            w_days = pycuda.cumath.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l]
            cuda_ops.cumsum_cuda(w_days)
            # fom in column format
            fom_sims = fom_sims_used[tenors_chosen.index(m), :]   # row vec
            mult_1 = np.float32(-0.5 * cv_m**2)
            mult_2 = np.float32(cv_m)
            col_vec = pycuda.cumath.exp(days_d * mult_1 + w_days * mult_2)
            # transpose is used
            spot_sims = cuda_ops.vtpv(fom_sims, col_vec, tm_ind='t',
                                      transpose_ind=True).transpose()
        else:  # no cuda
            w_days = np.cumsum(np.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l],
                               axis=1)
            # fom in column format
            fom_sims = fom_sims_used[m, :].reshape((len(fom_sims_used[tenors_chosen.index(m), :]), 1))
            spot_sims = np.transpose(fom_sims * np.exp(-0.5 * cv_m**2 * days + cv_m * w_days))
        return spot_sims

    def skew_params(self, asset_nb, C_vec, opt_mat_idx):
        """
        given the input coefficients and value of the factor X
        returns the skewed forward prices
        """
        cc1 = C_vec[0]  # coefficient at x^2
        cc2 = C_vec[1]  # coefficient at x^3
        cc3 = C_vec[2]  # coefficient at x^4

        # integrated volatility
        v = self.black_vol_current(asset_nb, opt_mat_idx) * \
            np.sqrt(self.option_tenors_list[asset_nb][opt_mat_idx])
        f0t = self.forward_curve_list[asset_nb][opt_mat_idx]

        a0 = (1. - cc1 * v**2 / 2. + cc3 * v**4 / 8.) * f0t
        a1 = (1. - cc2 * v**2 / 2.) * f0t
        a2 = (cc1 / 2. - cc3 * v**2 / 4.) * f0t
        a3 = (cc2 / 6.) * f0t
        a4 = (cc3 / 24.) * f0t

        return np.array([a0, a1, a2, a3, a4, v])

    def skew_tsf(self, asset_nb, X, C_vec, opt_mat_idx):
        """
        skew transformation of the entire curve for given X
        """
        [A0, A1, A2, A3, A4, V] = self.skew_params(asset_nb, C_vec, opt_mat_idx)
        return self.forward_curve_list[asset_nb] * (A0 + A1*X + A2*X**2 + A3*X**3 + A4*X**4)

    def skew_tsf_internal(self, asset_nb, X, opt_mat_idx):
        return self.skew_tsf(asset_nb, X, self.C_vec_list[asset_nb][opt_mat_idx], opt_mat_idx)

    def intra_curve_corr_calib(self, model_name_list):
        """
        inter-curve correlation calib.
        """
        for asset_1 in range(self.nb_assets):
            for asset_2 in range((asset_1+1), self.nb_assets):
                self.black_corr_intra_curves_calib(asset_1,asset_2)


mm_calib_file_actual = 'mobj/mm_calib.txt'
mm_calib_multi_file_actual = 'mobj/mm_calib_multiple.txt'
single_asset_mm_calib = config.work_dir + mm_calib_file_actual
multi_asset_mm_calib = config.work_dir + mm_calib_multi_file_actual


def read_mm_hash(multi_single_ind):
    """
    multi_single_ind ... indicator to load single or multiple asset models
    puts together a hash table of all already calibrated market models
    file has structure: COM, DATE_, NB_FWD (single asset)
                        (COM1, COM2), DATE_, (NFWD_1, NFWD_2)  for multiple assets
    """
    if multi_single_ind is 'single':
        mm_calib_file = single_asset_mm_calib
    else:
        mm_calib_file = multi_asset_mm_calib
    print "Reading prev. calibrated market models from", mm_calib_file

    mm_hash = dict()
    if os.stat(mm_calib_file)[6] != 0:  # is file empty
        f1 = open(mm_calib_file)
        f2 = csv.reader(f1)  # csv format
        for com, date_, nb_fwd_str in f2:
            if multi_single_ind is 'single':
                nb_fwd = int(nb_fwd_str)
                if com not in mm_hash.keys():
                    mm_hash[com] = dict()
                    mm_hash[com][date_] = nb_fwd
                else:
                    if date_ not in mm_hash[com].keys():
                        mm_hash[com][date_] = nb_fwd
                    else:
                        mm_hash[com][date_] = max(mm_hash[com][date_], nb_fwd)
            else:  # multiple models
                nb_fwd_l = tuple(nb_fwd_str.split('___'))  # in the form A1___A2
                nb_fwd_l_int = tuple([int(o) for o in nb_fwd_l])
                com_l = tuple(com.split('___'))
                if com_l not in mm_hash.keys():
                    mm_hash[com_l] = dict()
                    mm_hash[com_l][date_] = [nb_fwd_l_int]
                else:  # already have that com
                    if date_ not in mm_hash[com_l].keys():
                        mm_hash[com_l][date_] = [nb_fwd_l_int]
                    else:
                        if nb_fwd_l_int not in mm_hash[com_l][date_]:
                            mm_hash[com_l][date_].append(nb_fwd_l_int)
    return mm_hash

mm_hash = read_mm_hash('single')
mm_hash_multiple = read_mm_hash('multiple')


def write_mm_hash(multi_single_ind, mm_hash):
    """
    writes the hash tables into the file
    """
    if multi_single_ind is 'single':
        mm_calib_file = single_asset_mm_calib
    else:
        mm_calib_file = multi_asset_mm_calib

    f = open(mm_calib_file, 'w')
    for com in mm_hash.keys():
        for date_ in mm_hash[com].keys():
            if multi_single_ind is 'single':
                f.write(com + ',' + date_ + ',' + str(mm_hash[com][date_]) + '\n')
            else:
                # go through all fwd list
                string_comp = ''
                for asset in com:
                    string_comp += asset + '___'
                string_comp = string_comp[:-3]  # remove last ___
                string_comp += ',' + date_ + ','
                for fwd_pair in mm_hash[com][date_]:
                    string_final = string_comp
                    for nb_fwds in fwd_pair:
                        string_final += str(nb_fwds) + '___'
                    string_final = string_final[:-3]
                    string_final += '\n'
                    f.write(string_final)


def find_adj_tenors(com_nb,
                    adj_fwd_tenors_days,
                    adj_vol_tenors_days):

    if adj_vol_tenors_days is not None and adj_fwd_tenors_days is not None:
        if com_nb in adj_vol_tenors_days.keys():
            if com_nb in adj_fwd_tenors_days.keys():
                adj_fwd_tenors = adj_fwd_tenors_days[com_nb]
                adj_vol_tenors = adj_vol_tenors_days[com_nb]
            else:
                adj_fwd_tenors = adj_fwd_tenors_days[com_nb]
                adj_vol_tenors = None
        else:
            if com_nb in adj_fwd_tenors_days.keys():
                adj_fwd_tenors = adj_fwd_tenors_days[com_nb]
                adj_vol_tenors = None
            else:
                adj_fwd_tenors, adj_vol_tenors = None, None
    elif adj_vol_tenors_days is not None and adj_fwd_tenors_days is None:
        if com_nb in adj_vol_tenors_days.keys():
            adj_vol_tenors = adj_vol_tenors_days[com_nb]
            adj_fwd_tenors = None
        else:
            adj_fwd_tenors, adj_vol_tenors = None, None
    elif adj_vol_tenors_days is None and adj_fwd_tenors_days is not None:
        if com_nb in adj_fwd_tenors_days.keys():
            adj_fwd_tenors = adj_fwd_tenors_days[com_nb]
            adj_vol_tenors = None
        else:
            adj_fwd_tenors, adj_vol_tenors = None, None
    else:
        adj_fwd_tenors, adj_vol_tenors = None, None

    return adj_fwd_tenors, adj_vol_tenors


def find_mm(com, date_, fwd, mm_hash=mm_hash):
    """
    finds the market model for single asset
    """
    if com not in mm_hash.keys():
        return False
    else:
        if date_ not in mm_hash[com].keys():
            return False
        else:
            if fwd > mm_hash[com][date_]:
                return False
            else:
                return True


def find_mm_multiple(com_list_str, date_, fwd_list_str, mm_hash=mm_hash_multiple):
    """
    finds market model for multiple assets
    """
    if com_list_str not in mm_hash.keys():
        return False
    else:
        if date_ not in mm_hash[com_list_str].keys():
            return False
        else:
            if fwd_list_str not in mm_hash[com_list_str][date_]:
                return False
            else:
                return True


def update_mm_hash(multi_single_ind, mm_hash, new_mm):
    """

    """
    com, date_, nb_fwd = new_mm
    mm_hash_new = mm_hash
    if com not in mm_hash.keys():
        mm_hash_new[com] = dict()
        if multi_single_ind is 'single':
            mm_hash_new[com][date_] = nb_fwd
        else:
            mm_hash_new[com][date_] = [nb_fwd]
    else:
        if date_ not in mm_hash[com].keys():
            if multi_single_ind is 'single':
                mm_hash_new[com][date_] = nb_fwd
            else:
                mm_hash_new[com][date_] = [nb_fwd]
        else:
            if multi_single_ind is 'single':
                if nb_fwd > mm_hash_new[com][date_]:
                    mm_hash_new[com][date_] = nb_fwd
            else:
                if nb_fwd not in mm_hash_new[com][date_]:
                    mm_hash_new[com][date_].append(nb_fwd)
    return mm_hash_new


def mrds_calib(com, date_, nb_fwd, mm_hash=mm_hash,
               mm_hash_file=config.work_dir + mm_calib_file_actual,
               mt=True, model_ind='skew',
               cuda_ind=False):
    """
    calibrates the mrds object
    """
    mm_calibrated = find_mm(com, date_, nb_fwd, mm_hash)
    mobj_mm_beg = 'mobj/mm_'

    if mm_calibrated:
        mm_file = config.work_dir + mobj_mm_beg + str(com) + '_' + str(date_) + '_' + \
            str(mm_hash[com][date_]) + '.obj'
        mm = pickle.load(open(mm_file))
        mm.multi_thread_ind = mt
    else:
        mm_file = config.work_dir + mobj_mm_beg + str(com) + '_' + str(date_) + '_' + \
            str(nb_fwd) + '.obj'
        mm = mrds_calib_db(com, com, date_, nb_fwd, mt=mt, model_ind=model_ind,
                           cuda_ind=False)
        pickle.dump(mm, open(mm_file, 'wb'))
        mm_hash_new = update_mm_hash('single', mm_hash, [com, date_, nb_fwd])
        write_mm_hash('single', mm_hash_new)

    return mm


def mrds_calib_db(com_fwd, com_vol, date_, nb_fwd, mt=True,
                  model_ind='skew',
                  cuda_ind=False):
    mm = mrd_skew(1, date_, multi_thread_ind=mt, model_skew_ln_ind=model_ind,
                  cuda_ind=cuda_ind)
    mm.read_curve_vol_data_db(date_, 0, com_fwd, com_vol, sub_idx_rows=np.arange(nb_fwd))
    mm.read_discount_curve_db(date_)
    mm.read_model_config_db(0)
    mm.set_other_params(0)
    if mm.vol_surface_name_list[0] == 'ATM':
        mm.model_skew_ln_ind = 'ln_ln'
    mm.black_vol_calibration(0)
    if model_ind is 'skew':
        mm.calibrate_skew_params(0)
    mm.generate_large_corr_mat()
    return mm


def mrds_calib_multiple(com_l, date_, nb_fwd_l,
                        mm_hash=mm_hash_multiple,
                        model_ind='skew',
                        adj_fwd_tenors_days=None,
                        adj_vol_tenors_days=None,
                        multi_thread_ind=True,
                        cuda_ind=False):
    """
    calibrates the multiple models
    """
    nb_comm = len(com_l)
    assert(nb_comm == len(nb_fwd_l))
    com_str = reduce(lambda x, y: x+'___'+y, com_l)
    nb_fwd_str = reduce(lambda x, y: str(x)+'___'+str(y), nb_fwd_l)
    if nb_comm == 1:
        mm_calibrated = find_mm(com_l[0], date_, nb_fwd_l[0], mm_hash)
    else:
        mm_calibrated = find_mm_multiple(tuple(com_l), date_,
                                         tuple(nb_fwd_l), mm_hash)

    mm_file = config.work_dir + 'mobj/mm_' + str(com_str) + '_' + \
        str(date_) + '_' + str(nb_fwd_str) + '.obj'
    if mm_calibrated:
        mm = pickle.load(open(mm_file))
    else:
        mm = mrds_calib_db_multiple(com_l, com_l, date_, nb_fwd_l,
                                    model_ind=model_ind,
                                    adj_fwd_tenors_days=adj_fwd_tenors_days,
                                    adj_vol_tenors_days=adj_vol_tenors_days,
                                    mt=multi_thread_ind,
                                    cuda_ind=cuda_ind)
        pickle.dump(mm, open(mm_file, 'wb'))
        mm_hash_new = update_mm_hash('multiple', mm_hash,
                                     [tuple(com_l), date_, tuple(nb_fwd_l)])
        write_mm_hash('multiple', mm_hash_new)

    return mm


def mrds_calib_db_multiple(com_fwd_l, com_vol_l, date_, nb_fwd_l,
                           mt=True,
                           model_ind='skew',
                           adj_fwd_tenors_days=None,
                           adj_vol_tenors_days=None,
                           cuda_ind=False):
    """
    com_fwd_l ... list of com to calibrate
    col_vol_l ... list of vols for counterparted coms
    nb_fwd_l ... list of nb of forwards [12, 10, 12]
    mt ... multithreading indicator
    adj_fwd_tenors_days ... dict of {com_nb: nb_days_adj}
    """
    nb_comm = len(com_fwd_l)
    assert(nb_comm == len(com_vol_l) and nb_comm == len(nb_fwd_l)), \
        "Unequal lists for fwd, vol, nb_fwd"

    mm = mrd_skew(nb_comm, date_, multi_thread_ind=mt,
                  model_skew_ln_ind=model_ind,
                  cuda_ind=cuda_ind)
    mm.read_discount_curve_db(date_)
    for com_nb, (com_used, vol_used) in enumerate(zip(com_fwd_l, com_vol_l)):
        adj_fwd_tenors, adj_vol_tenors = find_adj_tenors(com_nb,
                                                         adj_fwd_tenors_days,
                                                         adj_vol_tenors_days)
        mm.read_curve_vol_data_db(date_, com_nb, com_used, vol_used,
                                  sub_idx_rows=np.arange(nb_fwd_l[com_nb]),
                                  adj_fwd_tenors_days=adj_fwd_tenors,
                                  adj_vol_tenors_days=adj_vol_tenors)
        mm.read_model_config_db(com_nb)
        mm.set_other_params(com_nb)
        mm.black_vol_calibration(com_nb)
        if model_ind is 'skew':
            mm.calibrate_skew_params(com_nb)
    mm.generate_large_corr_mat()
    return mm


def compute_partial_deltas (mm, pricer, params, nb_sim, subset_idx,
                            delta=0.01, seed=None, verbose=None):
    """
    # computes partial deltas for all assets
    # mm ... calibrated market model
    # pricer has to be in the form
    #   pricer (mo, params) where
    #     mo ... market object
    #     params ... parameters
    # nb_sim ... nb. of simulations
    # subset_idx ... for which futures compute the deltas
    # delta ... bump, default is the same as for structured desk
    # seed ... seed for random numbers
    # verbose ... prints additional things
    """
    #nb_sim = mm.simulated_curves[0].shape[2] # simulations from t
    deltas = mm._empty_list_fct (mm.nb_assets) # empty list of deltas

    # WOULD BE BETTER INTERPRETED AS A DICTIONARY CORRECT CORRECT CORRECT 
    for asset_nb in range(mm.nb_assets):
        deltas[asset_nb] = np.zeros(len(subset_idx))
        for idx in range(len(subset_idx)):
            if verbose is not None:
                print "Computing deltas for asset", asset_nb, "and fwd. idx.", subset_idx[idx]
                print "Original price of future's", subset_idx[idx], "for asset", asset_nb, "=", \
                    mm.forward_curve_list[asset_nb][subset_idx[idx]]
            mm.forward_curve_list[asset_nb][subset_idx[idx]] *= (1.0+delta)
            mm.simulate_curves(nb_sim, seed)
            deltas[asset_nb][idx] = pricer(mm, params)  # pricer with forward curves bumped
            mm.forward_curve_list[asset_nb][subset_idx[idx]] /= (1.0+delta)
            mm.simulate_curves(nb_sim, seed)
            price2 = pricer(mm, params)
            deltas[asset_nb][idx] -= price2
            if verbose is not None:
                print "Bumped price", mm.forward_curve_list[asset_nb][subset_idx[idx]]
                print "price 1:", deltas[asset_nb][idx]
                print "price 2:", price2
    return np.array(deltas)


def compute_partial_vegas(mm, pricer, params, nb_sim, subset_idx,
                          delta=0.001, seed=None, verbose='none'):
    
    vegas = mm._empty_list_fct(mm.nb_assets) # empty list of deltas
    for asset_nb in range(mm.nb_assets):
        vegas[asset_nb] = np.zeros(len(subset_idx))
        for idx in range(len(subset_idx)):
            # WRONG WRONG WRONG - HERE RECALIBRATION IS NEEDED
            mm.sigma_vec_list[asset_nb][subset_idx[idx]] += delta
            mm.simulate_curves(nb_sim, seed)
            vegas[asset_nb][idx] = pricer(mm, params)  # pricer with vol curves bumped
            mm.atm_vol_list[asset_nb][subset_idx[idx]] -= delta
            mm.simulate_curves(nb_sim, seed)
            vegas[asset_nb][idx] -= pricer(mm, params)
            vegas[asset_nb][idx] *= 100.0 # scaling
            if verbose is not None:
                print "Computing vegas for asset", asset_nb, "and fwd. idx.", subset_idx[idx]
                print "Bumped vol curve", mm.atm_vol_list[asset_nb]
                print "up", vegas[asset_nb][idx]
                print "down", pricer(mm, params)
    return vegas 


def compute_partial_gammas (mm, pricer, params, nb_sim, delta=0.01, set_seed=None, verbose=None):
    delta1 = compute_partial_deltas(mm, pricer, params, nb_sim, delta, set_seed, verbose)
    delta2 = compute_partial_deltas(mm, pricer, params, nb_sim, -delta, set_seed, verbose)
    return delta1 - delta2

