#
#   skew model for forward curves
#

import datetime
import numpy as np
import scipy
import scipy.stats
import scipy.interpolate  # spline package
from openopt         import NLP, NSP  # CHECK THIS HERE!!1
from logging         import Logger
from multiprocessing import Pool, cpu_count
from functools       import lru_cache
from typing          import List, Dict

# mrds imports
from mrds_maths    import ComMathsMixin
from mrds_defaults import ComSkewDefaultsMixin
from correlations  import corr_hyp_sec_mat
from near_corr     import near_corr_simple
from vols.vols     import getVolObject, Volatility  # , black_vol_inverse  TODO: ADD THIS black_vol_inverse back
from vols.vols_basic import black_vol_inverse
from discount      import read_discount_curve, read_discount_curve_quantlib
from forward_curve import FwdCurve
from quartic.quartic_cy import QuadRoots, CubicRoots, QuarticRoots
from opd.opd_avx   import skew_fom


logger = Logger(__name__)


class ComSkewError(Exception):
    """ Base class for ComSkew model exceptions.
    """

    pass


def opt_fct_skew_wrap(arg, **kwarg):
    """ Wrapper for the skew MRD model calibration function.
    """

    return ComSkew._opt_fct_skew(*arg, **kwarg)


class ComSkew(ComMathsMixin, ComSkewDefaultsMixin):
    """ Base class of the commodity skew market model.
    """

    NLP_SOLVER           = 'scipy_cobyla'
    MAX_ASSETS = 20  # maximum number of assets to be calibrated in the model

    _LRU_CACHE_SIZE_CALIB = 5    # lru for number of factors.
    _BETA_T_CACHE_SIZE    = 10   # cache size for beta_t
    _C_VEC_CACHE          = 100  # cache for C vector

    def __init__(self
                 , mkt_date      : datetime.date
                 , fwd_curves    : List[FwdCurve]
                 , vol_curves    : List[Volatility]
                 , calc_date     = None):

        """ Initialization of the skew model.

        :param mkt_date: market date
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, in case they are different than forward curves.
        :param calc_date: calculation date.
        """

        self._mktDate          = mkt_date
        self.__mkt_date_change = True  # indicator whether the market date has changed and everything needs to be recalculated.
        self._calcDate         = calc_date if calc_date else mkt_date
        self._com_fwd_curves   = fwd_curves
        self._com_vol_curves   = vol_curves
        nb_assets              = len(self._com_fwd_curves)

        # initial value of the calibrated params
        self.__C_vec = {fwd_curve.fwd_name: ComSkew._oneZeroZeroMatrix(len(fwd_curve.fwd_tenors), 3)
                         for fwd_curve in fwd_curves }

        # indicator functions - whether the values are updated
        # indicator function for the sigma, kappa calibration
        self.sigma_kappa_calib_indicator_list = np.repeat(False, nb_assets)
        self.skew_calib_indicator_list        = np.repeat(False, nb_assets)
        self.days_nb_const_ind    = False  # monthly dat numbers are constr.
        self.simulate_spot_rn_ind = False  # indic. for random number for spot sim.

        self.__black_vol_inverse_tol = 1e-4  # default value of the black vol inverse parameter

        # hashed values
        self.__com_curve_names      = None
        self.__discount_function    = None  # has for discount function
        self.__discount_function_ql = None  # placeholder for QuantLib discount function
        self.__factor_corr_mtx = dict()  # to keep track of the factor correlation matrices.
        self.__market_corr_mtx = dict()  # track of the market correlation matrix
        self.__complete_corr_mtx            = None  # complete correlation matrix hash
        self.__regenerate_complete_corr_mtx = True  # indicator whether to regenerate the complete corr. mtx.

    def __default_factor_corr_mat_fct( self
                                     , asset_1 : str
                                     , asset_2 : str
                                     , same_asset_corr = 0.98
                                     , diff_asset_corr = 0.96 ):
        """ Initial corr. matrix

        :param asset_1: first asset, e.g. 'WTI',
        :param asset_2: second asset, e.g. 'BRENT'
        """

        if asset_1 == asset_2:
            return corr_hyp_sec_mat(same_asset_corr, range(self.nb_factors_for_asset(asset_1)))

        # asset_1 != asset_2
        return diff_asset_corr * np.ones((self.nb_factors_for_asset(asset_1),
                                          self.nb_factors_for_asset(asset_2)))

    def __default_factor_corr_mat_fct_lb_ub( self
                                           , asset_1 : str
                                           , asset_2 : str
                                           , lb_ub_ind='ub' ):
        """ Sets the default factor correlation lower (lb) and upper (ub) bound between asset_1 and asset_2.

        :param asset_1: first asset for default correlation, e.g. 'WTI'
        :param asset_2: second asset, e.g. 'BRENT'
        :param lb_ub_ind: indicator whether it's upper bound 'ub' or lower bound 'lb'
        """

        lb_ub_fact = -0.999 if lb_ub_ind is 'lb' else 0.999

        if asset_1 == asset_2:
            tmp_1 = np.ones((self.nb_factors_for_asset[asset_1], self.nb_factors_for_asset[asset_1]))
            tmp_ut = np.triu(tmp_1, 1)
            tmp_lt = np.tril(tmp_1, -1)
            return tmp_1 - tmp_ut * 0.001 - tmp_lt * 0.001 if lb_ub_ind is 'ub'else \
                tmp_1 - tmp_ut * 1.999 - tmp_lt * 1.999

        return lb_ub_fact * np.ones((self.nb_factors_for_asset[asset_1], self.nb_factors_for_asset[asset_2]))

    @property
    def fwd_curves(self) -> List[FwdCurve]:
        """ Curve names for the commodity curves in the model.
        """

        return self._com_fwd_curves

    @property
    def fwd_curve_names(self) -> Dict[str, FwdCurve]:
        """ Memoizes the forward curve names.
        """
        if self.__com_curve_names:
            return self.__com_curve_names

        self.__com_curve_names = {fwd_curve.fwd_name: fwd_curve for fwd_curve in self.fwd_curves}
        return self.__com_curve_names

    @property
    def vol_curves(self) -> List[Volatility]:
        return self._com_vol_curves

    @property
    def vol_curve_names(self):
        if self.__vol_curve_names:
            return self.__vol_curve_names

        self.__vol_curve_names = {vol_curve.fwd_name: vol_curve for vol_curve in self.vol_curves}
        return self.__vol_curve_names

# TODO: INCLUDE THIS HERE
#        , 'corr_init': np.array([[1., 0.5], [0.5, 1.]])
#        , 'corr_lb': np.array([[1., -0.99], [-0.99, 1.]])
#        , 'corr_ub': np.array([[1., 0.99], [0.99, 1.]])} ):

    def _set_factor_corr_mat(self, asset_1, asset_2, new_corr_mtx = None, lb_ub_ind=None):
        """ Returns the factor correlation between assets 1 & 2. If new_corr_mtx is provided,
            set that as the correlation matrix between them.

        :param asset_1: first asset to get the correlation
        :param asset_2: second asset for the correlation
        :param new_corr_mtx: new matrix if you want it to be set up.
        """

        mtx_to_insert = new_corr_mtx if new_corr_mtx else (self.__default_factor_corr_mat_fct(asset_1, asset_2) if not lb_ub_ind else self.__default_factor_corr_mat_fct_lb_ub(asset_1, asset_2, lb_ub_ind=lb_ub_ind))

        if asset_1 not in self.__factor_corr_mtx:
            self.__factor_corr_mtx[asset_1] = {asset_2: mtx_to_insert}

        elif asset_2 not in self.__factor_corr_mtx[asset_1]:
            self.__factor_corr_mtx[asset_1][asset_2] = mtx_to_insert

        return self.__factor_corr_mtx[asset_1][asset_2]

    def _factor_corr_mat(self, asset_1 : str, asset_2 : str, lb_ub_ind = None):
        """ Gets the factor correlation matrix between assets 1 and 2.

        :param asset_1: first asset, e.g. 'WTI'
        :param asset_2: second asset, e.g. 'BRENT'
        :param lb_ub_ind: lower bound/upper bound indicator
        """

        if not lb_ub_ind:
            return self.__factor_corr_mtx['mid'][asset_1][asset_2]

        return self.__factor_corr_mtx[lb_ub_ind][asset_1][asset_2]

    def _market_corr(self, asset_1 : str, asset_2 : str, mkt_corr_mtx = None):  # TODO: THIS NEEDS TO BE CHANGED HERE
        """ Returns the market correlation, if mkt_corr_mtx is not provided.

        :param asset_1: first asset, e.g. 'WTI'
        :param asset_2: second asset, e.g. 'BRENT'
        :param mkt_corr_mtx: market correlation matrix
        """

        if mkt_corr_mtx:
            logger.info('Market correlation between assets {0} and {1} overwritten with {2}'.format(asset_1, asset_2, mkt_corr_mtx))

        mtx_to_insert = mkt_corr_mtx if mkt_corr_mtx else self.__default_factor_corr_mat_fct(asset_1, asset_2)

        if asset_1 not in self.__factor_corr_mtx:
            self.__market_corr_mtx[asset_1] = {asset_2: mtx_to_insert}

        elif asset_2 not in self.__factor_corr_mtx[asset_1]:
            self.__market_corr_mtx[asset_1][asset_2] = mtx_to_insert

        return self.__market_corr_mtx[asset_1][asset_2]

    @property
    def black_vol_inverse_tol(self):
        """ Tolerance for calibrating the vol matrix.
        """
        return self.__black_vol_inverse_tol

    @black_vol_inverse_tol.setter
    def black_vol_inverse_tol(self, new_inverse_tol):
        self.__black_vol_inverse_tol = new_inverse_tol

    @classmethod
    def from_db(cls
                , mkt_date   : datetime.date
                , fwd_curves : List[str] ):
        """ Constructs the class by reading forward and vol curves from the database.

        :param mkt_date: market date
        :param fwd_curves: list of forward curves to be read from database. (e.g. ['WTI', 'BRENT'])...
        """

        return cls( mkt_date
                  , [FwdCurve.from_db(mkt_date, fwdCurveName) for fwdCurveName in fwd_curves]
                  , [getVolObject(fwd_curve, mkt_date) for fwd_curve in fwd_curves])  # TODO: VOL CURVE HAS TO BE REFACTORED

    @property
    def mkt_date(self) -> datetime.date:
        return self._mktDate

    @mkt_date.setter
    def mkt_date(self, new_mkt_date : datetime.date):
        """ Sets the new market date, updates all the curves accordingly.
        """

        self._mktDate = new_mkt_date
        self.__mkt_date_change = True  # TODO: MAKE THE MODEL DEPENDENT ON THIS!!!

    @property
    def calc_date(self) -> datetime.date:
        return self._calcDate

    @calc_date.setter
    def calc_date(self, new_calc_date : datetime.date):
        """ Sets the new calculation date, updates all the curves accordingly.

        :param new_calc_date: new caluclation date.
        """

        self._calcDate = new_calc_date

    def nb_factors_for_asset(self, asset : str) -> int:
        """ Number of factors per asset, placeholder perhaps for some other function.
        """

        if asset not in self.fwd_curve_names:
            raise ComSkewError('Requested curve {0} not in list of curves: {1}'.format(asset, self.fwd_curve_names.keys()))

        return 2

    def simulation_times( self
                        , sim_times : [np.ndarray, List[datetime.date], List[str], List[float] ]
                        , dcf = 365.25 ):
        """ Returns the simulation times from the list of sim_times.

        :param sim_times: new simulation times in one of the accepted formats
        :param dcf: day-count-factor
        """

        if type(sim_times) == np.ndarray:
            return sim_times, [self.mkt_date + datetime.timedelta(int(np.round(stf * dcf)))
                               for stf in sim_times]

        if (type(sim_times) == list) and (type(sim_times[0]) == datetime.datetime):
            sim_times_normalized = np.array([(st - self.mkt_date).days / 365. for st in sim_times])
            return sim_times_normalized, sim_times_normalized

        if (type(sim_times) == list) and (type(sim_times[0]) == float):
            sim_times_normalized = np.array(sim_times)
            return sim_times_normalized, [self.mkt_date + datetime.timedelta(int(np.round(stf * 365.))) for stf in sim_times_normalized]

    def update_market_date(self, new_market_date : datetime.date) -> None:
        """ Updates the date to the new market date, and updates the curves and vols accordingly.

        :param new_market_date: new date that one wants to set.
        """

        for comCurves in self._com_fwd_curves:
            comCurves.mkt_date = new_market_date

        for volCurve in self._com_vol_curves:
            volCurve.mkt_date = new_market_date

    @property
    def _discount_function(self):
        """ Returns the discount function for the market date.
        """

        if self.__discount_function:
            return self.__discount_function

        self.__discount_function = read_discount_curve(self.mkt_date)

        return self.__discount_function

    @property
    def _discount_function_ql(self):
        """ Returns the Quantlib discount function for the market date.
        """

        if self.__discount_function_ql:
            return self.__discount_function_ql

        self.__discount_function_ql = read_discount_curve_quantlib(self.mkt_date)

        return self.__discount_function_ql

    def DF( self
          , fwd_time : [float, datetime.date]
          , dcf = 365.25 ):
        """ Discount from self.mkt_date to fwd_time. Using basic discount curve.

        :param fwd_time: future time to discount to. can be '20140101', ...
        :param dcf: day-count factor.
        """

        if (type(fwd_time) is np.double) or (type(fwd_time) is float):
            time_diff = fwd_time

        elif isinstance(fwd_time, datetime.datetime):
            time_diff = self.__difference_to_market_date(fwd_time, dcf=dcf)

        else:
            raise ComSkewError('fwd_time given in function DF is not of form [str, double, float]')

        return scipy.interpolate.splev(time_diff, self._discount_function)

    def DF_ql(self, fwd_time :[datetime.date, float], dcf = 365.25 ):
        """ Discount from self.mkt_date to t

        :param fwd_time: future time to discount to. can be '20140101', ...
        :param dcf: day-count factor.
        """

        return self.__discount_function_ql.discount(fwd_time)

    @staticmethod
    def _construct_corr(mtx_size, theta_vector):
        """ Constructs and upper triangular matrix from a vector theta_vector, first row is from the rho matrix.

        :param mtx_size: matrix size.
        :param theta_vector:
        """

        utm = np.triu(np.ones((mtx_size, mtx_size)))  # upper triangular matrix
        utm_diag_ones = np.diag(np.diag(utm))
        utm = utm - utm_diag_ones
        utm[utm == 1] = theta_vector

        return utm + utm.transpose() + utm_diag_ones

    def _construct_corr_asset(self, asset_nb, theta_vector):
        return self.__class__._construct_corr(self.nb_factors_for_asset(asset_nb), theta_vector)

    def __difference_to_market_date(self, fwd_date : datetime.date, dcf=365.25) -> float:
        """ Computes the difference to market date given the discount factor.
        """

        return (fwd_date - self.mkt_date).days / dcf

    def __fwd_square_vol (self
                          , asset       : str
                          , kappa_vec   : np.array
                          , sigma_vec   : np.array
                          , corr_matrix : np.array
                          , fwd_date_2    : datetime.date
                          , fwd_date_1  : datetime.date ):
        """ Computes forward integrated square vol (function V) from fwd_date_1 to fwd_date_2.

        :param asset: asset to be considered. (e.g. 'WTI')
        :param kappa_vec: vec of kappas
        :param sigma_vec: vector of sigmas
        :param corr_matrix: correlation matrix for asset
        :param fwd_date_1: start of forward volatility computation
        :param fwd_date_2: end of forward vol computation
        """

        nb_factors = self.nb_factors_for_asset(asset)

        sigma_vec_row = sigma_vec.reshape((1, nb_factors))
        sigma_vec_col = sigma_vec.reshape((nb_factors, 1))
        kappa_vec_row = kappa_vec.reshape((1, nb_factors))
        kappa_vec_col = kappa_vec.reshape((nb_factors, 1))

        cross_1 = self._betaT(asset, [fwd_date_2]) ** 2 * sigma_vec_row * corr_matrix * sigma_vec_col
        cross_2 = kappa_vec_row + kappa_vec_col

        time_to_fwd_date_2 = self.__difference_to_market_date(fwd_date_2)
        time_to_fwd_date_1 = self.__difference_to_market_date(fwd_date_1)

        return np.sum(cross_1 * (np.exp(-cross_2 * ( time_to_fwd_date_2 - time_to_fwd_date_1)) -
                                 np.exp(-cross_2 * time_to_fwd_date_2)) / cross_2)

    def __V_one_factor( self
                      , asset      : str
                      , factor_nb  : int
                      , fwd_date   : datetime.date
                      , t_0        : float
                      , t_1        : float ) -> float:
        """ Computes integrated volatility V only for one factor (factor_nb).

        :param asset: asset to consider (e.g. 'WTI')
        :param factor_nb: factor to consider (0, 1,...)
        :param fwd_date: forward date
        :param t_0: integrated volatility start time (float)
        :param t_1: integrated vol end time (float)
        """

        kappa = self._kappa_vec(asset)[factor_nb]
        sigma = self._sigma_vec(asset)[factor_nb]
        beta  = self._betaT(asset, fwd_date)

        if kappa == 0.:
            return beta**2 * sigma**2 * (t_1 - t_0)

        return beta**2 * sigma**2 / (2. * kappa) * \
               np.exp(-2. * kappa * self.__difference_to_market_date(fwd_date)) * \
               (np.exp(2. * kappa * t_1) - np.exp(2. * kappa * t_0))

    def _V_cross_factor(self
                        , asset_nb : str
                        , factor_1 : int
                        , factor_2 : int
                        , fwd_date_1 : datetime.date
                        , fwd_date_2 : datetime.date
                        , t_0      : float
                        , t_1      : float):
        """  Computes cross integrated vol. V between factors factor_1 and factor_2.

        :param asset: asset to consider (e.g. 'WTI')
        :param factor_1: first factor to consider (0, 1,...)
        :param factor_2: second factor to consider (0,1,...)
        :param fwd_date_1: forward date_1
        :param fwd_date_2: forward date 2
        :param t_0: integrated volatility start time (float)
        :param t_1: integrated vol end time (float)
        """

        kappa_1 = self._kappa_vec(asset_nb)[factor_1]
        kappa_2 = self._kappa_vec(asset_nb)[factor_2]
        kappa_12 = kappa_1 + kappa_2
        sigma_1 = self._sigma_vec(asset_nb)[factor_1]
        sigma_2 = self._sigma_vec(asset_nb)[factor_2]
        rho_12 = self._factor_corr_mat(asset_nb, asset_nb)[factor_1, factor_2]
        beta_1 = self._betaT(asset_nb, fwd_date_1)
        beta_2 = self._betaT(asset_nb, fwd_date_2)

        if kappa_12 == 0.:
            return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 * (t_1 - t_0)

        return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 / kappa_12 * \
               (np.exp(-kappa_1 * (self.__difference_to_market_date(fwd_date_1) - t_1) - kappa_2 * (T_2-t_1)) -
                np.exp(-kappa_1 * (self.__difference_to_market_date(fwd_date_2) - t_0) - kappa_2 * (T_2-t_0)))

    def black_vol(self
                  , asset      : str
                  , kappa_vec
                  , sigma_vec
                  , corr_matrix
                  , fwd_date    : datetime.date):
        """ Computes the model black vol until fwd_date
        """

        return np.sqrt(self.__fwd_square_vol( asset
                                             , kappa_vec
                                             , sigma_vec
                                             , corr_matrix
                                             , fwd_date  # TODO: THE NEXT ARGUMENT IS WRONG
                                             , fwd_date) / self.__difference_to_market_date(fwd_date))

    def black_vol_current(self, asset_nb : str, fwd_date : datetime.date) -> float:
        """ Computes black vol until option maturity for the given model parameters.

        :param asset_nb: asset number (e.g. 'WTI')
        :param fwd_date: forward date
        :returns: black volatility for the model TODO: REWRITE THESE DESCRIPTIONS
        """

        # return self.black_vol(asset
        #                       , self._kappa_vec(asset)
        #                       , self._sigma_vec(asset)
        #                       , self._factor_corr_mat(asset, asset)
        #                       , fwd_date)

        return np.sqrt(self.__fwd_square_vol( asset_nb
                                            , self._kappa_vec(asset_nb)
                                            , self._sigma_vec(asset_nb)
                                            , self._factor_corr_mat(asset_nb, asset_nb)
                                            , self.mkt_date
                                            , fwd_date) / self.__difference_to_market_date(fwd_date))

    def __V_current( self
                   , asset    : str
                   , fwd_contract_date : datetime.date
                   , fwd_date          : datetime.date ):
        """ Function V(t) for selected asset.

        :param asset: asset that you want the forward square vol of.
        :param fwd_date: forward contract one wants to compute the square vol of.
        :param t: Describe here this parameter
        """

        return self.__fwd_square_vol(asset
                                     , self._kappa_vec(asset)
                                     , self._sigma_vec(asset)
                                     , self._factor_corr_mat(asset, asset)
                                     , fwd_date
                                     , fwd_date)

    def __option_tenor_for_fwd_tenor(self, asset: str, fwd_tenor : datetime.date) -> datetime.date :
        """ Returns the option tenor for a forward tenor.
            TODO: IMPROVE LATER.

        :param asset: asset we are requesting.
        :param fwd_tenor: a forward tenor for which the option tenor we are requesting.
        """

        return fwd_tenor

    def __distance_model_market_black_vol( self
                                         , asset_nb
                                         , kappa_vec
                                         , sigma_vec
                                         , rho_vec):
        """ Distance between model & market black volatility, used for calibration.

       :param asset_nb: asset to consider, e.g. 'WTI'
       :param kappa_vec: kappa vector to calibrate
       :param sigma_vec: sigma vector to calibrate for asset asset
       :param rho_vec: correlation _vector_ to calibrate
        """

        return np.sum((np.array([self.black_vol( asset_nb
                                               , kappa_vec
                                               , sigma_vec
                                               , self._construct_corr_asset(asset_nb, rho_vec)
                                               , fwd_date )
                                 for fwd_date in self.fwd_curves[asset_nb].fwd_tenors])
                       - self.vol_curves[asset_nb].atmVol() ) ** 2)  # TODO: FIX THIS atm_vol

    @lru_cache(maxsize=MAX_ASSETS)
    def __kappa_sigma_rho(self, asset : str):
        """ Calibrates kappa and sigma and rho parameters of the log-normal part of the model.

        :param asset: asset to be calibrated
        """

        nbf = self.nb_factors_for_asset(asset)

        # extracting the upper triangular part of the correlation matrix 
        fcm_init = self._factor_corr_mat(asset, asset)
        fcm_lb   = self._factor_corr_mat(asset, asset, lb_ub_ind='lb')
        fcm_ub   = self._factor_corr_mat(asset, asset, lb_ub_ind='ub')

        # optimization run
        return NLP( lambda kappa_sigma_rho_vec: self.__distance_model_market_black_vol( asset
                                                                                      , kappa_sigma_rho_vec[0:nbf]
                                                                                      , kappa_sigma_rho_vec[nbf:(2*nbf)]
                                                                                      , kappa_sigma_rho_vec[(2*nbf):] )
                       , np.concatenate([ self._kappa_default(nbf, 'init')
                                        , self._sigma_default(nbf, 'init')
                                        , np.triu(fcm_init, 1)[np.triu(fcm_init, 1) != 0] ])
                       , lb = np.concatenate([self._kappa_default(nbf, 'lb'),
                                              self._sigma_default(nbf, 'lb'),
                                              np.triu(fcm_lb, 1)[np.triu(fcm_lb, 1) != 0] ])
                       , ub = np.concatenate([self._kappa_default(nbf, 'ub'),
                                              self._sigma_default(nbf, 'ub'),
                                              np.triu(fcm_ub, 1)[np.triu(fcm_ub, 1) != 0] ])
                       , iprint = -1 )\
                       .solve(self.__class__.NLP_SOLVER)

    def _kappa_vec(self, asset : str) -> np.ndarray:
        """ Holds the kappa vector for a particular asset.

        :param asset: asset to compute the kappa vector of.
        """

        return self.__kappa_sigma_rho(asset).xf[0:self.nb_factors_for_asset(asset)]  # number of factors

    def _sigma_vec(self, asset : str):
        """ Calibrated sigma vector, depends on the __kappa_sigma_rho above.

        :param asset: asset to calculate sigma vector over.
        """

        nbf = self.nb_factors_for_asset(asset)  # number of factors

        return self.__kappa_sigma_rho(asset).xf[nbf:(2 * nbf)]

    @lru_cache(maxsize=_LRU_CACHE_SIZE_CALIB)
    def __factorCorrMatList(self, asset : str):
        """ Factor correlation matrix
            TODO: FIX THIS!!! WHAT DOES THIS SERVE!!!

        """

        return self._construct_corr_asset(asset, self.__kappa_sigma_rho(asset).xf[(2 * self.nb_factors_for_asset(asset)):])

    @lru_cache(maxsize=_BETA_T_CACHE_SIZE)
    def _betaT( self
              , asset : str
              , tenors = None ):
        """ Adjusts beta_T so that the atm vol is fitted perfectly.
            (assuming that kappa, sigma, rho has already been calibrated).
            The results are memoized.

        :param asset: name of the asset calibrated (e.g. 'WTI')
        :param tenors: list of tenors for which the beta is calibrated ( normally in form: List[datetime.date])
        """

        tenors_used = tenors if tenors else self.vol_curves[asset].tenors

        return self.vol_curves[asset].atmVol(tenors_used) / \
               np.array([ self.black_vol( asset
                                        , self._kappa_vec(asset)
                                        , self._sigma_vec(asset)
                                        , self._factor_corr_mat(asset, asset)
                                        , tenor)
                         for tenor in tenors_used])

    def __black_corr_within_curve(self, asset : str, ind_1 : int, ind_2 : int):
        """ Cummulative correlation between the ind_1-th and the ind_2-th future's contract
            up to the option time of the smallest of the two contracts

        :param asset: curve asset
        :param ind_1: contract ind_1 on the curve
        :param ind_2: contract ind_2 on the curve
        """

        # TODO: opt_mat is WRONG
        opt_mat = self.__option_tenor_for_fwd_tenor(asset, np.min(ind_1, ind_2))  # opt_mat until the smallest one
        corr = self._factor_corr_mat(asset, asset)  # correlation matrix
        kv = self._kappa_vec(asset)
        sv = self._sigma_vec(asset)
        ft = self.fwd_curve_names[asset].fwd_values  # forward vector
        nbf = self.nb_factors_for_asset(asset)

        a = np.array([corr[ind_1, ind_2] * sv[factor_nb_1] * sv[factor_nb_2] *
                      np.product(self._betaT[asset]) *
                      (np.exp(- kv[factor_nb_1] * (ft[ind_1] - opt_mat) -
                              kv[factor_nb_2] * (ft[ind_2] - opt_mat) ) -
                       np.exp(- kv[factor_nb_1] * ft[ind_1] -
                              kv[factor_nb_2] * ft[ind_2])) /
                      (kv[factor_nb_1] + kv[factor_nb_2])
                      for factor_nb_1 in range(nbf)
                      for factor_nb_2 in range(nbf)])

        return (np.exp(np.sum(a)) - 1.) / self.black_vol(asset, kv, sv, ind_1) / \
               self.black_vol(asset, kv, sv, ind_2)  # covariance divided by 2 standard deviations

    def black_corr_intra_curves( self
                               , model_corr_mtx
                               , curve_1 : str
                               , curve_2 : str
                               , tenor_1 : datetime.date
                               , tenor_2 : datetime.date ) -> np.double:
        """

        :param curve_1: name of the first curve
        :param curve_2: name of second commodity curve
        :param tenor_1: first tenor
        :param tenor_2: second tenor
        """

        t_1 = self.__option_tenor_for_fwd_tenor(curve_1, tenor_1)
        t_2 = self.__option_tenor_for_fwd_tenor(curve_2, tenor_2)
        opt_mat = t_1 if  t_1 <= t_2 else t_2  # opt_mat until the smallest one
        kv1 = self._kappa_vec(curve_1)
        kv2 = self._kappa_vec(curve_2)

        bv1 = np.sqrt(self.__V_current(curve_1, tenor_1, opt_mat))  # square of integrated variance
        bv2 = np.sqrt(self.__V_current(curve_2, tenor_2, opt_mat))

        return sum([model_corr_mtx[factor_nb_1, factor_nb_2] *
                    self._sigma_vec(curve_1)[factor_nb_1] * self._sigma_vec(curve_2)[factor_nb_2] *
                    self._betaT(curve_1, tenor_1) * self._betaT(curve_2, tenor_2) *
                    np.exp(- kv1[factor_nb_1] * self.__difference_to_market_date(tenor_1)
                           - kv2[factor_nb_2] * self.__difference_to_market_date(tenor_2)) *
                    (np.exp((kv1[factor_nb_1] + kv2[factor_nb_2]) * opt_mat) - 1) /
                    (kv1[factor_nb_1] + kv2[factor_nb_2] )
                    for factor_nb_1 in range(self.nb_factors_for_asset(curve_1))
                    for factor_nb_2 in range(self.nb_factors_for_asset(curve_2))]) / (bv1 * bv2)

    def __black_corr_intra_curves_factors( self
                                         , model_corr_mtx
                                         , curve_1     : str
                                         , curve_2     : str
                                         , tenor_1     : datetime.date
                                         , tenor_2     : datetime.date
                                         , factor_nb_1 : int
                                         , factor_nb_2 : int
                                         , opt_mat ):
        """  Same as function above (black_corr_intra_curves), but the factors are exposed

        :param curve_1: forward curve 1
        :param curve_2: forward curve 2
        :param tenor_1: tenor index for tenor on curve 1
        :param tenor_2: tenor on curve 2
        :param factor_nb_1: factor on curve 1
        :param factor_nb_2: factor on curve 2
        :param opt_mat: until what maturity this is computed TODO: WHAT DOES THIS MAKE SENSE??
        """

        kv1 = self._kappa_vec(curve_1)
        kv2 = self._kappa_vec(curve_2)
        sv1 = self._sigma_vec(curve_1)
        sv2 = self._sigma_vec(curve_2)
        bv1 = np.sqrt(self.__V_one_factor(curve_1, factor_nb_1, tenor_1, 0., opt_mat))
        bv2 = np.sqrt(self.__V_one_factor(curve_2, factor_nb_2, tenor_2, 0., opt_mat))

        return model_corr_mtx[factor_nb_1, factor_nb_2] * \
               sv1[factor_nb_1] * sv2[factor_nb_2] * \
               self._betaT(curve_1, tenor_1) * self._betaT(curve_2, tenor_2) * \
               np.exp(- kv1[factor_nb_1] * self.__difference_to_market_date(tenor_1) - kv2[factor_nb_2] * self.__difference_to_market_date(tenor_2)) * \
               (np.exp((kv1[factor_nb_1] + kv2[factor_nb_2]) * opt_mat) - 1.) / \
               (kv1[factor_nb_1] + kv2[factor_nb_2]) / (bv1 * bv2)

    def __black_corr_intra_curves_calib( self
                                       , curve_1 : str
                                       , curve_2 : str
                                       , solver = 'scipy_cobyla' ):
        """ Calibrates the correlations between different curves.

        :param curve_1: curve 1 to for correlation calibration.
        :param curve_2: curve 2 for calibration.
        :param solver: solver to use in the OpenOpt
        """

        black_corr_intra_curve_vector = lambda model_corr_mtx, curve_1, curve_2, corr_len: \
            np.array([self.black_corr_intra_curves( model_corr_mtx
                                                  , curve_1
                                                  , curve_2
                                                  , tenor
                                                  , tenor )
                      for tenor in range(corr_len)])
            
        black_corr_intra_curve_vector_optim = lambda model_corr_mtx, curve_1, curve_2, corr_len: \
            scipy.linalg.norm(black_corr_intra_curve_vector(model_corr_mtx, curve_1, curve_2, corr_len) -
                              self._market_corr[curve_1][curve_2])

        corr_len_real = len(self._market_corr(curve_1, curve_2))
        curve_1_nb_fact = self.nb_factors_for_asset(curve_1)
        curve_2_nb_fact = self.nb_factors_for_asset(curve_2)
        optim_pr = NSP(lambda corr_mtx_ravel: black_corr_intra_curve_vector_optim(corr_mtx_ravel.reshape ((curve_1_nb_fact,curve_2_nb_fact)),
                                                                                  curve_1, curve_2, corr_len_real),
                       self._factor_corr_mat(curve_1, curve_2).ravel(),
                       lb = self._factor_corr_mat(curve_1, curve_2, lb_ub_ind='lb').ravel(),
                       ub = self._factor_corr_mat(curve_1, curve_2, lb_ub_ind='ub').ravel())\
                      .solve(solver)

        correlation_matrix = np.array(optim_pr.xf).reshape((curve_1_nb_fact, curve_2_nb_fact))

        self.__factorCorrMatList[curve_1][curve_2] = correlation_matrix
        self.__factorCorrMatList[curve_2][curve_1] = correlation_matrix

        return self.__factorCorrMatList[curve_1][curve_2]

    def __deltas_to_strikes(self
                            , asset : str
                            , tenor_date     : datetime.date
                            , delta_vec_list : np.array ) -> np.array:
        """
        Converts deltas to strikes for particular asset and tenor.

        :param asset: commodity asset to compute
        :param tenor_date: tenor considered.
        :returns: a vector of deltas from the strikes given in self.delta_vec_list
        """

        integrated_vol = self.vol_curve_names[asset].atm_vol(tenor_date) * \
                         np.sqrt(self.__difference_to_market_date(self.__option_tenor_for_fwd_tenor(asset, tenor_date)))

        return np.exp((scipy.stats.norm.ppf(delta_vec_list) - 0.5 * integrated_vol ) * integrated_vol) * \
               self.fwd_curve_names[asset].fwd_value(tenor_date)

    @staticmethod
    def __integr_analy( real_roots_tsf
                      , nb_real_roots  : int
                      , A0
                      , A1
                      , A2
                      , A3
                      , A4
                      , V ):
        """ Integrate the polynomial between the roots, used for option calibration.
            Computes E[ p(x)_+ ]  TODO: CHECK HERE.
            A_0 x**4 + A_1 x**3 + A_2 x**2 + A_3 x + A_4

        :param real_roots_tsf:
        :param nb_real_roots: number of real roots

        """

        # TODO: THIS IS NOT CORRECT PROBABLY
        Asigma = np.array([A0, A1, A2, A3, A4]) * np.array([1., V, V ** 2, V ** 3, V ** 4])  # A multiplied by sigmas

        if nb_real_roots == 0:  # integrate polynomial function over whole of real axis
            if A4 > 0 or (A4 == 0 and A2 > 0) or (A4 == 0 and A2 == 0 and A0 > 0):
                return Asigma[0] + Asigma[2] + 3. * Asigma[4]

            return 0.

        if nb_real_roots == 1:
            if A3 > 0:
                return np.sum(ComSkew._trunc_normal_below(real_roots_tsf[0]) * Asigma)

            return np.sum(ComSkew._trunc_normal_above(real_roots_tsf[0]) * Asigma)

        if nb_real_roots in [2, 3]:  # integrate over 2 intervals
            if A4 > 0:
                return np.sum(ComSkew._trunc_normal_above(real_roots_tsf[0]) * Asigma) + \
                       np.sum(ComSkew._trunc_normal_below(real_roots_tsf[1]) * Asigma)

            if A4 < 0.:
                return np.sum(ComSkew._trunc_normal_interval(real_roots_tsf[0], real_roots_tsf[1]) * Asigma)

            if A4 == 0. and A3 != 0.:
                if A3 > 0.:
                    return np.sum(ComSkew._trunc_normal_interval(real_roots_tsf[0], real_roots_tsf[1]) * Asigma) + \
                           np.sum(ComSkew._trunc_normal_below(real_roots_tsf[2]) * Asigma)
                # A3 < 0
                return np.sum(ComSkew._trunc_normal_above(real_roots_tsf[0]) * Asigma) + \
                       np.sum(ComSkew._trunc_normal_interval(real_roots_tsf[1], real_roots_tsf[2]) * Asigma)
            if A4 == 0. and A3 == 0.:
                if A2 < 0.:
                    return np.sum(ComSkew._trunc_normal_interval(real_roots_tsf[0], real_roots_tsf[1]) * Asigma)

                return np.sum(ComSkew._trunc_normal_above(real_roots_tsf[0]) * Asigma) + \
                       np.sum(ComSkew._trunc_normal_below(real_roots_tsf[1]) * Asigma)

        # elif nb_real_roots == 4:  # integrate over 3 intervals
        if A4 > 0:
            return np.sum(ComSkew._trunc_normal_above(real_roots_tsf[0]) * Asigma) + \
                   np.sum(ComSkew._trunc_normal_below(real_roots_tsf[3]) * Asigma) + \
                   np.sum(ComSkew._trunc_normal_interval(real_roots_tsf[1], real_roots_tsf[2]) * Asigma)
        # A4 < 0
        return np.sum(ComSkew._trunc_normal_interval(real_roots_tsf[0], real_roots_tsf[1]) * Asigma) + \
               np.sum(ComSkew._trunc_normal_interval(real_roots_tsf[2], real_roots_tsf[3]) * Asigma)

    def __integr_num(self, A_V, call_put_ind, strike):
        """
        Integrate numerically between the roots of the polynomials.
        IMPORTANT: For testing purposes only, in prod. use the code in integr_analy.

        """

        from scipy.integrate import quad

        A0, A1, A2, A3, A4, V = ComSkew.__unpack_params(A_V, call_put_ind, strike)

        return quad(lambda x: np.max([A0 + A1 * V * x + A2 * V**2 * x**2 +
                                      A3 * V**3 * x**3 + A4 * V**4 * x**4, 0.]) / \
                                      np.sqrt(2. * np.pi) * np.exp(- x**2 / 2.)
                   , -np.inf
                   , np.inf)[0]

    @staticmethod
    def __unpack_params(A_V, call_put_ind : int, strike : float) -> tuple:
        """ Unpacks the parameters A, V from A_V.
            V ... integrated volatility.

        :param A_V: vector of A0, A1, A2, A3, A4, V

        """

        A0, A1, A2, A3, A4, V = A_V

        if call_put_ind == 1:  # call
            return (A0 - strike, A1, A2, A3, A4, V)
        # put
        return (strike - A0, - A1, -A2, -A3, -A4, V)

    def polynomial_european( self
                           , asset_nb : str
                           , C_vec
                           , opt_mat_idx
                           , strike
                           , call_put_ind
                           , ttm          : float
                           , debug_mode = False ) -> float:
        """
        value of european call option in skew model with strike
        call_put_ind ... 1 for call, -1 for put

        :param asset_nb: asset to consider, e.g. 'WTI'.
        :param C_vec: vector of skew parameters
        :param opt_mat_idx:
        :param strike: strike of the option TODO
        :param call_put_ind: indicator whether this is a call or a put.
        :param ttm: time to maturity of the option.
        :param debug_mode: debug mode, computes the value numerically

        """

        # obtaining the coefficients
        A0, A1, A2, A3, A4, V = ComSkew.__unpack_params( self.skew_params(asset_nb, C_vec, opt_mat_idx)
                                                       , call_put_ind
                                                       , strike)

        if debug_mode:
            poly_roots = np.sort(np.poly1d([A4, A3, A2, A1, A0]).roots)
        else:
            if A4 == 0. and A3 == 0. and A2 == 0.:
                poly_roots = [-A0 / A1]
            elif A4 == 0. and A3 == 0.:
                poly_roots = np.sort(QuadRoots(np.array([A2, A1, A0])))
            elif A4 == 0.:
                poly_roots = np.sort(CubicRoots(np.array([A3, A2, A1, A0])))
            elif np.abs(A4) < 1e-6:
                poly_roots = np.sort(np.poly1d([A4, A3, A2, A1, A0]).roots)
            else:
                poly_roots = np.sort(QuarticRoots(np.array([A4, A3, A2, A1, A0])))

        real_roots = poly_roots[poly_roots == poly_roots.real].real  # real roots only


        #if debug_mode:  # debug, selects the numeric approach
        #    return disc_fact * self.__integr_num(Asigma, call_put_ind, strike)
        #else:  # production mode

        return self.DF(ttm) * \
                   ComSkew.__integr_analy( real_roots / V
                                         , np.sum(poly_roots == poly_roots.real)  # number of real roots
                                         , A0
                                         , A1
                                         , A2
                                         , A3
                                         , A4
                                         , V )

    def __model_vol_surface(self, asset : str, C_vec, fwd_date : datetime.date):
        """ Computes model vols for asset, C_vec, fwd_idx.

        :param asset: commodity considered.
        :param C_vec: skew vector
        """

        strikes = self.__deltas_to_strikes(asset, fwd_date)  # TODO: fix this part
        cp_ind = np.array([1 if (strike >= self.fwd_curve_names[asset].fwd_value(fwd_date)) else -1 for strike in strikes])

        return np.array([black_vol_inverse(self.fwd_curve_names[asset].fwd_value(fwd_date)
                                           , strike
                                           , opt_price
                                           , self.__option_tenor_for_fwd_tenor(asset, fwd_date)
                                           , self.DF(self.__difference_to_market_date(self.__option_tenor_for_fwd_tenor(asset, fwd_date)))
                                           , cp
                                           , self.black_vol_inverse_tol)
                         for opt_price, strike, cp in zip( [self.polynomial_european(asset, C_vec, fwd_date, strike, cp)
                                                            for strike, cp in zip(strikes, cp_ind)]
                                                         , strikes
                                                         , cp_ind ) ] )

    def _opt_fct_skew( self
                     , asset : str
                     , fwd_dates : List[datetime.date]):
        """ Optimization function to minimize over the fwd_dates.

        :param asset: asset for which the skew function is calibrates, e.g. ('WTI')
        :param fwd_dates: list of forward dates for calibration of the skew.
        """

        # penalize the calibrated funtion for values of C where positive forward prices.
        # penalization level is 10000
        #    imp_vol_vec_model - self.vol_surface_list[asset][fwd_idx, :]
        return NLP( lambda C_vec: scipy.linalg.norm(self.__model_vol_surface(asset, C_vec, fwd_dates) -
                                                    self.vol_curve_names[asset].getVolForDate(fwd_dates))
                  , np.array([1., 0., 0.])  # TODO: THIS HAS TO BE IMPROVED
                  , iprint = -1 )\
                  .solve(self.__class__.NLP_SOLVER).xf

    @lru_cache(maxsize=_C_VEC_CACHE)
    def __c_vec_list( self
                    , asset_nb  : int
                    , fwd_dates : List[datetime.date]
                    , multi_thread_ind = False):
        """ Returns the skew parameters for asset. If done the first time, it calibrates the parameters,
            otherwise it returns the stored value.

        :param asset_nb: the asset nb. to calibrate, such as 'wti'
        :param fwd_dates: forward dates for which to calibrate C
        :param multi_thread_ind: indicator whether to use multiple threads
        """

        if not multi_thread_ind:
            return np.array(self._opt_fct_skew(asset_nb, fwd_dates))

        # using multithreading
        with Pool(processes=cpu_count()) as pool:
            curr_nb_tenors = len(fwd_dates)
            C = pool.map(opt_fct_skew_wrap,
                         zip([self] * curr_nb_tenors,
                             [asset_nb] * curr_nb_tenors,
                             range(curr_nb_tenors)))
            return np.array(C)

    def _complete_corr_mtx(self, nb_steps=300) -> np.ndarray:
        """  Generates the factor correlation matrix from a list of list of corr. matrices
        gathered in __factorCorrMatList

        :param nb_steps: number of steps to converge to the correlation matrix, maybe 30 steps is enough
        :returns: None, just sets the self.__completeCorrMat
        """

        if not self.__regenerate_complete_corr_mtx:
            return self.__complete_corr_mtx

        # starting to generate the complete correlation matrix

        total_nb_factors = sum([self.nb_factors_for_asset(asset) for asset in self.fwd_curves])
        self.__complete_corr_mtx = np.zeros((total_nb_factors, total_nb_factors))

        # sets the large correlation building blocks
        for asset_1 in self.fwd_curves:
            for asset_2 in self.fwd_curves:
                self.__complete_corr_mtx[self.__factor_positions(asset_1), self.__factor_positions(asset_2)] = self._ # self.__factor_corr_mtx(asset_1, asset_2)

        # find the closest matrix that is positive semidefinite
        self.__complete_corr_mtx = near_corr_simple(self.__complete_corr_mtx, nb_steps)
        while not (np.linalg.eig(self.__complete_corr_mtx)[0] > 0.).all():
            d1, v1 = np.linalg.eig(self.__complete_corr_mtx)
            d1p = np.diag(np.maximum(d1, 1.e-16))
            self.__complete_corr_mtx = np.dot(v1, np.dot(d1p, v1.transpose()))

        return self.__complete_corr_mtx

    def _var_covar_mtx( self
                      , asset_nb : str
                      , fwd_idx
                      , i
                      , j
                      , t_idx
                      , sim_times ):
        """
        Generate covar mtx, part of LN simulation, used in simulate_curves.

        :param asset_nb: asset number considered
        """

        t_prev = 0. if t_idx == 0 else sim_times[t_idx - 1]
        t_next = sim_times[t_idx]

        if i == j:
            return self.__V_one_factor(asset_nb, i, fwd_idx, t_prev, t_next)

        return self._V_cross_factor(asset_nb, i, j, fwd_idx, fwd_idx, t_prev, t_next)

    @staticmethod
    def __simulate_std_normal( nb_factors     : int
                             , corr_mtx       : np.array
                             , nb_simulations : int ):
        """ Simulates the standard normal random variables with specified correlation

        :param corr_mtx: correlation matrix, a nb_factors x nb_factors matrix.
        :param nb_simulations: number of simulations from the factors.
        """

        return np.random.multivariate_normal( np.zeros(nb_factors)
                                            , corr_mtx
                                            , size = nb_simulations )

    def simulate_curves(self
                        , nb_simulations   : int
                        , simulation_times : List[datetime.date]
                        , tenor_list       : List[datetime.date]
                        , set_seed         = None) -> np.array :
        """ Simulate all curves for desired simulation times on either cpu or cuda.
            Simulation times have to be given.

        Generates a 3-dimensional array
        0-th dimension: asset
        1-st dimension: simulation times
        2-nd dimension: curve
        3-rd dimension: repeats of the curve

        :param nb_simulations: self. explanatory
        :param simulation_times: simulation times for the forwards.
        :param tenor_list: list of tenors which to simulate
        :param set_seed: seed, if needed, can be left to None
        :returns: a matrix of simulated paths
        """

        np.random.seed(set_seed)

        simulated_curves = {}
        fwd_c_col = {}

        for com_curve in self.fwd_curves:
            sim_curves_shape = (len(simulation_times), len(tenor_list), nb_simulations)
            simulated_curves[com_curve] = np.empty(sim_curves_shape)  #  if not cuda_ind else gpa.zeros(sim_curves_shape, dtype=rn_type)
            fwd_c_col[com_curve] = self.fwd_curves[com_curve].fwd_value(tenor_list)
            simulated_curves[com_curve][0, :, :] = fwd_c_col


        X      = [np.zeros((len(fwd_c_col[comCurve]), nb_simulations)) for comCurve in self.fwd_curves]
        X_prev = [np.empty ((len(fwd_c_col[comCurve]), nb_simulations)) for comCurve in self.fwd_curves]

        nb_factors = np.sum(self.nb_factors_for_asset)

        # looping over time steps
        #   simulates ln process, basis for skew as well
        #   t_i ... idx of sim_time
        #   fact_sum ... factors of the individual assets
        nb_time_steps = len(simulation_times)
        complete_corr_mtx = self.__complete_corr_mtx()
        for t_i in range(nb_time_steps):
            simulated_rn = self.__class__.__simulate_std_normal( nb_factors
                                                               , complete_corr_mtx
                                                               , nb_simulations )

            for fwd_curve in self.fwd_curves:
                asset = fwd_curve.fwd_name
                nb_factors_asset = self.nb_factors_for_asset(asset)
                old_cov_mat = complete_corr_mtx[self.__factor_positions(asset), self.__factor_positions(asset)]
                tenor_used = tenor_list[asset]
                old_chol_inv = np.linalg.inv(np.linalg.cholesky(old_cov_mat))
                sims_Z_unit = np.dot(old_chol_inv, simulated_rn[:, self.__factor_positions(asset)].transpose())

                for tenor_idx, tenor_nb in enumerate(tenor_used):
                    # prepare cov mtx
                    cov_chol = np.linalg.cholesky(np.array([[self._var_covar_mtx(asset, tenor_nb, i, j, t_i, simulation_times)
                                                             for j in range(nb_factors_asset)]
                                                            for i in range(nb_factors_asset)]))

                    delta_X = np.sum(np.dot(cov_chol, sims_Z_unit), axis=0)
                    # quadratic variation of delta_X, also q_v = V_u
                    qv = np.sum([[self._V_cross_factor( asset
                                                      , factor_1
                                                      , factor_2
                                                      , tenor_nb
                                                      , tenor_nb
                                                      , 0. if t_i == 0 else simulation_times[t_i - 1]
                                                      , simulation_times[t_i])
                                  for factor_1 in range(nb_factors_asset)]
                                 for factor_2 in range(nb_factors_asset)])

                    X_prev[asset][tenor_idx, :] = X[asset][tenor_idx, :]
                    X[asset][tenor_idx, :] = X_prev[asset][tenor_idx, :] + delta_X

                    # F_res = F_u * (1. + X_u + 0.5 * c1 * (X_u**2 - V_u) +
                    #                c2 * (X_u**3 - 3. * X_u * V_u) / 6. +
                    #                c3 * (X_u**4 - 6. * V_u * X_u**2 + 3. * V_u**2) / 24.)
                    # self.simulated_curves[asset][t_i, tenor_idx, :] = F_res
                    c1, c2, c3 =  self.__C_vec[asset][tenor_nb, :]
                    skew_fom( self._com_fwd_curves[asset][tenor_nb]
                            , X[asset][tenor_idx, :]  # delta_X
                            , 0.5 * c1
                            , qv  # V_u, quadratic variation
                            , c2/6.
                            , c3/24.
                            , simulated_curves[asset][t_i, tenor_idx, :]
                            , nb_simulations )

        return simulated_curves

    def simulate_1nb( self, nb_simulations : int
                    , simulation_times = None
                    , set_seed         = None ) -> Dict[str, np.ndarray]:
        """ Simulate the 1NB (rolling) contract.

        # TODO: FIX THIS HERE!!
        generates a 3-dimensional array:
          0-th dimension: asset
          1-st dimension: simulation times
          2-rd dimension: repeats of the curve

        :param nb_simulations: number of simulations to simulate.
        :param simulation_times: times when to simulate curves, if None TODO: WHAT THEN???
        :param set_seed: set the seed for simulations.
        """

        assert simulation_times[-1] > self.fwd_curves[0][-1], \
            'Last simulation time is larger than the largest forward tenor.'

        simulated_curves = self.simulate_curves(nb_simulations, simulation_times, set_seed)
        new_simulated_curves = {}

        for fwd_curve in self.fwd_curves:
            asset = fwd_curve.fwd_name
            new_simulated_curves[asset] = np.empty((len(simulation_times), nb_simulations))
            for t_idx, t_i in enumerate(simulation_times):
                current_tenor_nb = np.sum(self.fwd_curves[asset] <= t_i)
                new_simulated_curves[asset][t_i, :] = simulated_curves[asset][t_i, current_tenor_nb, :]

        return new_simulated_curves

    @lru_cache(maxsize=20)  # TODO: THIS IS NOT RIGHT HERE!!!
    def __factor_positions(self, asset : str) -> slice:
        """ Returns factor positions in a matrix for asset.

        :param asset: asset for which factor positions are obtained, e.g. 'WTI'
        """

        # TODO: THIS BELOW IS WRONG, IMPROVE THIS PART
        cums = np.cumsum(self.nb_factors_for_asset)
        fact_sum = np.zeros(len(cums)+1, dtype=np.int)
        fact_sum[1:(len(cums)+1)] = cums

        return slice(fact_sum[asset_nb], fact_sum[asset_nb+1])

    def simulate_curves_fom(self
                            , asset          : str
                            , nb_simulations : int
                            , sim_times      : List[datetime.date]
                            , tenors_list = None
                            , seed = None
                            , model = 'skew'):
        """ Simulate first of month (fom) curves.

        generates a list of 2 dim arrays:
           1-st dim: tenor
           2-nd dim: simulation

        :param asset: asset for which to simulate, e.g. 'WTI'
        :param nb_simulations: nb. of simulations
        :param sim_times: simulation times
        :param tenors_list: list of tenors to simulate TODO: LOTS TO DISCUSS HERE!!!
        :param model: which model: 'ln' or 'skew', default 'skew'
        """

        np.random.seed(seed=seed)
        sim_times = self.vol_curves[asset].vol_tenors if not tenors_list else self.vol_curves[asset].vol_tenors[tenors_list]
        complete_corr_mtx = self._complete_corr_mtx()

        # looping over tenors
        #    t_i ... idx of sim_time (also tenor)
        #    fact_sum ... factors of the individual assets
        sim_fom = np.empty((len(tenors_list), nb_simulations))  # nb_tenors x nb_simulations
        for tenor_idx, tenor_date in enumerate(tenors_list):
            tenor_nb = tenor_date
            t_curr = sim_times[tenor_idx]
            F_curr = self.fwd_curves[asset][tenor_nb]
            nb_factors = np.sum(self.nb_factors_for_asset)  # total nb. of factors
            simulated_rn = np.random.multivariate_normal( np.zeros(nb_factors)
                                                        , complete_corr_mtx
                                                        , size=nb_simulations)
            nb_factors_asset = self.nb_factors_for_asset[asset]
            new_cov_mat = np.array([[self._var_covar_mtx(self, asset, tenor_nb, i, j, tenor_idx, sim_times)
                                    for j in range(nb_factors_asset)]
                                    for i in range(nb_factors_asset)])

            new_chol = np.linalg.cholesky(new_cov_mat)
            old_cov_mat = complete_corr_mtx[self.__factor_positions(asset), self.__factor_positions(asset)]
            old_chol = np.linalg.cholesky(old_cov_mat)

            sims_Z = simulated_rn[:, self.__factor_positions(asset)].transpose()
            sims_Z_unit = np.dot(np.linalg.inv(old_chol), sims_Z)
            delta_X = np.sum(np.dot(new_chol, sims_Z_unit), axis=0)
            qv = np.sum([[self._V_cross_factor(asset, factor_1, factor_2, tenor_nb, tenor_nb, 0., t_curr)
                          for factor_1 in range(nb_factors_asset)]
                         for factor_2 in range(nb_factors_asset)])

            if model == 'ln':
                sim_fom[tenor_idx, :] = F_curr * np.exp(delta_X - 0.5 * qv)
            else:  # skew model
                c0, c1, c2 = self.__C_vec[asset][tenor_nb, :]
                # sim_fom[t_i, :] = F_curr * \
                #                    (1. + delta_X + 0.5 * c1 * (delta_X**2 - qv) +
                #                    c2 * (delta_X**3 - 3. * delta_X * qv) / 6. +
                #                    c3 * (delta_X**4 - 6. * qv * delta_X**2 + 3. * qv**2) / 24.)
                skew_fom( F_curr
                        , delta_X
                        , 0.5 * c0
                        , qv
                        , c1/6.
                        , c2/24.
                        , sim_fom[tenor_idx, :]
                        , nb_simulations )

        return sim_fom

    def skew_params(self
                    , asset    : str
                    , C_vec    : np.array
                    , fwd_date  : datetime.date) -> tuple :
        """ Given the C parameters, returns the parameters for the option value computation using the polynomial approach.

        :param asset: number of the asset
        :param C_vec: vector of calibrated skew parameters, [c0, c1, c2]
        :returns: a tuple of A0, A1, A2, A3, A4, V
        """

        cc1, cc2, cc3 = C_vec
        # integrated volatility
        v = self.black_vol_current(asset, fwd_date) * \
            np.sqrt(self.__difference_to_market_date(self.__option_tenor_for_fwd_tenor(asset, fwd_date)))
        f0t = self.fwd_curves[asset].fwd_value(fwd_date)

        return ( (1. - cc1 * v**2 / 2. + cc3 * v**4 / 8.) * f0t
               , (1. - cc2 * v**2 / 2.) * f0t
               , (cc1 / 2. - cc3 * v**2 / 4.) * f0t
               , (cc2 / 6.) * f0t
               , (cc3 / 24.) * f0t
               , v )


class ComSkewChecks(ComSkew):
    """ Commodity skew model w/ checks for the correlation, more for debugging.

    """

    def check_black_vol_calib(self, asset : str, reportingDiff=1e-2) -> None:
        """
        Checks the black vol calibration, logs the results if the calibration failed.

        :param asset: asset to be checked, e.g. 'WTI'
        :param reportingDiff: difference between model and market vols to be reported.
        """

        model_atm_vols = np.array([self.black_vol(asset
                                                  , self._kappa_vec(asset)
                                                  , self._sigma_vec(asset)
                                                  , self.__factor_corr_mtx(asset, asset), fwd)
                                   for fwd in range(self.forward_curve_len[asset])])

        diff = scipy.linalg.norm(model_atm_vols - self.atm_vol_list[asset])

        if diff > reportingDiff:
            logger.info('Calibration of ATM vols for asset nb. {0} FAILED. Diff= {1}'.format(asset, str(diff)))

        logger.debug('Calibration of ATM vols for asset nb. {0} succeeded. Diff = {1}'.format(asset, str(diff)))

    def __default_corr_mat(self, asset : str, exp_nb : float) -> np.ndarray:
        """ Constructs the default correlation matrix
        the closer exp_nb is to 0, the more singular the matrix is
        and the correlation between forwards is closer to 1
        (does not need optimization)

        :param asset: asset in the model to be considered ('WTI')
        :param exp_nb: correlation number for the correlation matrix.

        """

        nb_tenors = len(self.fwd_curves[asset].fwd_tenors)

        # TODO: THIS IS WRONG - MUST DEPEND ON tenor distance.
        return np.array([ [ np.exp(-(np.abs(j-i)*exp_nb))
                            for i in range(nb_tenors) ]
                          for j in range(nb_tenors) ] )
