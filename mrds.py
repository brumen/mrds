#
#   skew model for forward curves
#

import datetime
import numpy as np
import scipy
import scipy.stats
import scipy.interpolate  # spline package
from openopt         import NLP, NSP
from logging         import Logger
from multiprocessing import Pool, cpu_count
from functools       import lru_cache
from typing          import List, Dict, Tuple

# mrds imports
from mrds_maths    import ComMathsMixin
from mrds_defaults import ComSkewDefaultsMixin
from correlations  import corr_hyp_sec_mat
from near_corr     import near_corr_simple
from vols.vols     import Volatility
from vols.vols_get import get_vol_object
from vols.vols_basic import black_vol_inverse

from forward_curve       import FwdCurve
from discount            import DiscountCurve
from quartic.quartic_cy  import QuadRoots, CubicRoots, QuarticRoots
from tolling.opd.opd_avx import skew_fom


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
                 , discount_curve = None
                 , calc_date      = None ):

        """ Initialization of the skew model.

        :param mkt_date: market date
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, in case they are different than forward curves.
        :param discount_curve: discount curve, a function of fwd_date, returns lambda fwd_date: discount(mkt_date, fwd_date)
        :param calc_date: calculation date.
        """

        self._mktDate          = mkt_date
        self._calcDate         = calc_date if calc_date else mkt_date
        self._com_fwd_curves   = fwd_curves
        self._com_vol_curves   = vol_curves
        self._discount_curve   = discount_curve if discount_curve else DiscountCurve.discount_function(mkt_date)
        nb_assets              = len(self._com_fwd_curves)

        # initial value of the calibrated params
        self.__C_vec = None

        # indicator functions - whether the values are updated
        # indicator function for the sigma, kappa calibration
        self.sigma_kappa_calib_indicator_list = np.repeat(False, nb_assets)
        self.skew_calib_indicator_list        = np.repeat(False, nb_assets)
        self.days_nb_const_ind    = False  # monthly dat numbers are constr.
        self.simulate_spot_rn_ind = False  # indic. for random number for spot sim.

        self.__black_vol_inverse_tol = 1e-4  # default value of the black vol inverse parameter

        # hashed values
        self.__com_curve_names      = None
        self.__vol_curve_names      = None
        self.__factor_corr_mtx = dict()  # to keep track of the factor correlation matrices.
        self.__market_corr_mtx = dict()  # track of the market correlation matrix
        self.__complete_corr_mtx            = None  # complete correlation matrix hash
        self.__regenerate_complete_corr_mtx = True  # indicator whether to regenerate the complete corr. mtx.

    @classmethod
    def from_db(cls
                , mkt_date   : datetime.date
                , fwd_curves : List[str] ):
        """ Constructs the class by reading forward and vol curves from the database.

        :param mkt_date: market date
        :param fwd_curves: list of forward curves to be read from database. (e.g. ['WTI', 'BRENT'])...
        """

        return cls( mkt_date
                  , [FwdCurve.from_db(mkt_date, fwd_curve) for fwd_curve in fwd_curves]
                  , [get_vol_object(fwd_curve, mkt_date)   for fwd_curve in fwd_curves])

    @property
    def mkt_date(self) -> datetime.date:
        return self._mktDate

    @mkt_date.setter
    def mkt_date(self, new_mkt_date : datetime.date):
        """ Sets the new market date, updates all the curves accordingly.
        """

        self._mktDate = new_mkt_date

        # TODO: All the things that should change on market date
        for comCurves in self._com_fwd_curves:
            comCurves.mkt_date = new_mkt_date

        for volCurve in self._com_vol_curves:
            volCurve.mkt_date = new_mkt_date

    @property
    def calc_date(self) -> datetime.date:
        return self._calcDate

    @calc_date.setter
    def calc_date(self, new_calc_date : datetime.date):
        """ Sets the new calculation date, updates all the curves accordingly.

        :param new_calc_date: new caluclation date.
        """

        self._calcDate = new_calc_date

    @property
    def fwd_curves(self) -> List[FwdCurve]:
        """ Curve names for the commodity curves in the model.
        """

        return self._com_fwd_curves

    def fwd_curve_names(self, asset : str) -> FwdCurve:
        """ Memoizes the forward curve names and returns the forward curve for a particular asset.

        :param asset: asset for which the forward curve is computed, e.g. ('WTI')
        """

        if self.__com_curve_names:
            return self.__com_curve_names[asset]

        self.__com_curve_names = {fwd_curve.fwd_name: fwd_curve for fwd_curve in self.fwd_curves}
        return self.__com_curve_names[asset]

    @property
    def vol_curves(self) -> List[Volatility]:
        return self._com_vol_curves

    def vol_curve_names(self, asset : str) -> Volatility:
        """ Same as vol curves, but it produces a dictionary where keys are assets.

        :param asset: asset for which vol you want to obtain (e.g. 'WTI')
        """

        if self.__vol_curve_names:
            return self.__vol_curve_names[asset]

        self.__vol_curve_names = {vol_curve.com_name: vol_curve for vol_curve in self.vol_curves}

        return self.__vol_curve_names[asset]

    @property
    def black_vol_inverse_tol(self):
        """ Tolerance for calibrating the vol matrix.
        """
        return self.__black_vol_inverse_tol

    @black_vol_inverse_tol.setter
    def black_vol_inverse_tol(self, new_inverse_tol):
        self.__black_vol_inverse_tol = new_inverse_tol

    def _c_vec(self, asset : str, fwd_date : datetime.date) -> np.array:
        """ Returns the C vector (skew vector) for the asset and forward date.

        :param asset: asset for which C vector
        :param fwd_date: forward date.
        """

        if not self.__C_vec:
            logger.info('Calibrating the model for asset {0} and fwd point {1}'.format(asset, fwd_date))
            self.__C_vec = {asset: {fwd_date: self._opt_fct_skew(asset, [fwd_date])[fwd_date]}}
            return self.__C_vec[asset][fwd_date]

        if asset not in self.__C_vec:
            logger.info('Calibrating the model for asset {0} and fwd point {1}'.format(asset, fwd_date))
            self.__C_vec[asset] = {fwd_date: self._opt_fct_skew(asset, [fwd_date])[fwd_date]}
            return self.__C_vec[asset][fwd_date]

        if fwd_date not in self.__C_vec[asset]:
            logger.info('Calibrating the model for asset {0} and fwd point {1}'.format(asset, fwd_date))
            self.__C_vec[asset][fwd_date] = self._opt_fct_skew(asset, [fwd_date])[fwd_date]
            return self.__C_vec[asset][fwd_date]

        return self.__C_vec[asset][fwd_date]  # already computed, just return it.

    def _set_c_vec(self, asset : str, fwd_date : datetime.date, c_vec_value : np.array) -> np.array:
        """ Sets the c_vec value.

        :param asset: asset for which C vector
        :param fwd_date: forward date.
        :param c_vec_value: new value for c_vector
        """

        if not self.__C_vec:
            self.__C_vec = {asset: {fwd_date: c_vec_value}}

        if asset not in self.__C_vec:
            self.__C_vec[asset] = {fwd_date: c_vec_value}

        if fwd_date not in self.__C_vec[asset]:
            self.__C_vec[asset][fwd_date] = c_vec_value

    def __default_factor_corr_mat( self
                                 , asset_1 : str
                                 , asset_2 : str
                                 , same_asset_corr = 0.98
                                 , diff_asset_corr = 0.96 ):
        """ Default correlation matrix between asset_1 and asset_2.

        :param asset_1: first asset, e.g. 'WTI',
        :param asset_2: second asset, e.g. 'BRENT'
        :param same_asset_corr: default correlation between tenors on the same curve.
        :param diff_asset_corr: default correlation between different assets
        """

        if asset_1 == asset_2:
            return corr_hyp_sec_mat(same_asset_corr, range(self.nb_factors_for_asset(asset_1)))

        # asset_1 != asset_2
        return diff_asset_corr * np.ones((self.nb_factors_for_asset(asset_1),
                                          self.nb_factors_for_asset(asset_2)))

    def __factor_corr_mat_lb_ub(self
                                , asset_1 : str
                                , asset_2 : str
                                , lb_ub_ind = 'ub') -> np.ndarray:
        """ Sets the default factor correlation lower (lb) and upper (ub) bound between asset_1 and asset_2.

        :param asset_1: first asset for default correlation, e.g. 'WTI'
        :param asset_2: second asset, e.g. 'BRENT'
        :param lb_ub_ind: indicator whether it's upper bound 'ub' or lower bound 'lb'
        """
        # TODO: INCLUDE THIS HERE
        #        , 'corr_init': np.array([[1., 0.5], [0.5, 1.]])
        #        , 'corr_lb': np.array([[1., -0.99], [-0.99, 1.]])
        #        , 'corr_ub': np.array([[1., 0.99], [0.99, 1.]])} ):

        lb_ub_fact = -0.999 if lb_ub_ind is 'lb' else 0.999

        if asset_1 == asset_2:
            tmp_1 = np.ones((self.nb_factors_for_asset(asset_1), self.nb_factors_for_asset(asset_1)))
            tmp_ut = np.triu(tmp_1, 1)
            tmp_lt = np.tril(tmp_1, -1)
            return tmp_1 - tmp_ut * 0.001 - tmp_lt * 0.001 if lb_ub_ind is 'ub'else \
                tmp_1 - tmp_ut * 1.999 - tmp_lt * 1.999

        return lb_ub_fact * np.ones((self.nb_factors_for_asset(asset_1), self.nb_factors_for_asset(asset_2)))

    def _factor_corr_mat(self
                         , asset_1 : str
                         , asset_2 : str
                         , new_corr_mtx = None
                         , lb_ub_ind    = None) -> np.ndarray:
        """ Returns the factor correlation matrix between assets 1 & 2. If new_corr_mtx is provided,
            set that as the correlation matrix between them.

        :param asset_1: first asset to get the correlation
        :param asset_2: second asset for the correlation
        :param new_corr_mtx: new matrix if you want it to be set up.
        :param lb_ub_ind: indicator whether upper bound or lower bound is set. Options: 'lb', 'ub'
        """

        mtx_to_insert = new_corr_mtx if new_corr_mtx else (self.__default_factor_corr_mat(asset_1, asset_2) if not lb_ub_ind else self.__factor_corr_mat_lb_ub(asset_1, asset_2, lb_ub_ind=lb_ub_ind))

        if asset_1 not in self.__factor_corr_mtx:
            self.__factor_corr_mtx[asset_1] = {asset_2: mtx_to_insert}

        elif asset_2 not in self.__factor_corr_mtx[asset_1]:
            self.__factor_corr_mtx[asset_1][asset_2] = mtx_to_insert

        return self.__factor_corr_mtx[asset_1][asset_2]

    def __factor_corr_mat_multiple(self, assets : List[str]) -> np.ndarray:
        """ Returns the factor correlation matrix for all assets in assets

        :param assets: list of assets for which the correlation should be returned.
        :returns: a complete correlation matrix for these assets
        """

        # first determine the size of the matrix
        nb_factors_by_asset = [self.nb_factors_for_asset(asset) for asset in assets]
        total_nb_factors = sum(nb_factors_by_asset)

        total_corr_mat = np.empty((total_nb_factors, total_nb_factors))
        # now set the individual matrices
        for asset_1 in assets:
            for asset_2 in assets:
                asset_1_in_list = assets.index(asset_1)
                asset_2_in_list = assets.index(asset_2)
                start_ind_1 = sum(nb_factors_by_asset[:asset_1_in_list])
                end_ind_1   = start_ind_1 + nb_factors_by_asset[asset_1_in_list]
                start_ind_2 = sum(nb_factors_by_asset[:asset_2_in_list])
                end_ind_2   = start_ind_2 + nb_factors_by_asset[asset_2_in_list]
                total_corr_mat[start_ind_1:end_ind_1, start_ind_2:end_ind_2] = self._factor_corr_mat(asset_1, asset_2)

        return total_corr_mat

    def _market_corr(self, asset_1 : str, asset_2 : str, mkt_corr_mtx = None):  # TODO: THIS NEEDS TO BE CHANGED HERE
        """ Returns the market correlation, if mkt_corr_mtx is not provided.

        :param asset_1: first asset, e.g. 'WTI'
        :param asset_2: second asset, e.g. 'BRENT'
        :param mkt_corr_mtx: market correlation matrix
        """

        if mkt_corr_mtx:
            logger.info('Market correlation between assets {0} and {1} overwritten with {2}'.format(asset_1, asset_2, mkt_corr_mtx))

        mtx_to_insert = mkt_corr_mtx if mkt_corr_mtx else self.__default_factor_corr_mat(asset_1, asset_2)

        if asset_1 not in self.__factor_corr_mtx:
            self.__market_corr_mtx[asset_1] = {asset_2: mtx_to_insert}

        elif asset_2 not in self.__factor_corr_mtx[asset_1]:
            self.__market_corr_mtx[asset_1][asset_2] = mtx_to_insert

        return self.__market_corr_mtx[asset_1][asset_2]

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

    def __black_corr_within_curve(self
                                  , asset: str
                                  , fwd_date_1: datetime.date
                                  , fwd_date_2: datetime.date) -> float:
        """ Cumulative correlation between the fwd_date_1 and fwd_date_2 points on the forward curve.
            up to the option time of the smallest of the two contracts

        :param asset: curve asset
        :param fwd_date_1: first contract on the curve
        :param fwd_date_2: second contract on the curve
        """

        opt_mat = self.__option_tenor_for_fwd_tenor(asset,
                                                    fwd_date_1 if fwd_date_1 <= fwd_date_2 else fwd_date_2)  # opt_mat until the smallest one
        num_fwd_opt_date_1 = self.__numerical_dist_to_mktdate(fwd_date_1) - self.__numerical_dist_to_mktdate(opt_mat)
        num_fwd_opt_date_2 = self.__numerical_dist_to_mktdate(fwd_date_2) - self.__numerical_dist_to_mktdate(opt_mat)
        num_fwd_date_1 = self.__numerical_dist_to_mktdate(fwd_date_1)
        num_fwd_date_2 = self.__numerical_dist_to_mktdate(fwd_date_2)
        corr = self._factor_corr_mat(asset, asset)  # correlation matrix
        kv = self._kappa_vec(asset)
        sv = self._sigma_vec(asset)
        nbf = self.nb_factors_for_asset(asset)

        a = np.array([corr[factor_nb_1, factor_nb_1] * sv[factor_nb_1] * sv[factor_nb_2] *
                      np.product(self._beta_T[asset]) *
                      (np.exp(- kv[factor_nb_1] * num_fwd_opt_date_1 - kv[factor_nb_2] * num_fwd_opt_date_2) -
                       np.exp(- kv[factor_nb_1] * num_fwd_date_1 - kv[factor_nb_2] * num_fwd_date_2)) /
                      (kv[factor_nb_1] + kv[factor_nb_2])
                      for factor_nb_1 in range(nbf)
                      for factor_nb_2 in range(nbf)])

        # covariance divided by 2 standard deviations
        return (np.exp(np.sum(a)) - 1.) / self.black_vol(asset, fwd_date_1, kv, sv, corr) / \
               self.black_vol(asset, fwd_date_2, kv, sv, corr)

    def __black_corr_intra_curves(self
                                  , model_corr_mtx: np.ndarray
                                  , curve_1: str
                                  , curve_2: str
                                  , tenor_1: datetime.date
                                  , tenor_2: datetime.date) -> float:
        """ Returns the black correlation between different curves and different forward points.

        :param model_corr_mtx: correlation matrix between factors of the model, same for all tenors, e.g. [2x2 matrix]
        :param curve_1: name of the first curve
        :param curve_2: name of second commodity curve
        :param tenor_1: first tenor
        :param tenor_2: second tenor
        """

        t_1 = self.__option_tenor_for_fwd_tenor(curve_1, tenor_1)
        t_2 = self.__option_tenor_for_fwd_tenor(curve_2, tenor_2)
        opt_mat = t_1 if t_1 <= t_2 else t_2  # opt_mat until the smallest of the two tenors
        kv1 = self._kappa_vec(curve_1)
        kv2 = self._kappa_vec(curve_2)

        bv1 = np.sqrt(self.__V_current(curve_1, tenor_1, opt_mat))  # square of integrated variance
        bv2 = np.sqrt(self.__V_current(curve_2, tenor_2, opt_mat))

        return sum([model_corr_mtx[factor_nb_1, factor_nb_2] *
                    self._sigma_vec(curve_1)[factor_nb_1] * self._sigma_vec(curve_2)[factor_nb_2] *
                    self._beta_T(curve_1, tenor_1) * self._beta_T(curve_2, tenor_2) *
                    np.exp(- kv1[factor_nb_1] * self.__difference_to_market_date(tenor_1)
                           - kv2[factor_nb_2] * self.__difference_to_market_date(tenor_2)) *
                    (np.exp((kv1[factor_nb_1] + kv2[factor_nb_2]) * opt_mat) - 1) /
                    (kv1[factor_nb_1] + kv2[factor_nb_2])
                    for factor_nb_1 in range(self.nb_factors_for_asset(curve_1))
                    for factor_nb_2 in range(self.nb_factors_for_asset(curve_2))]) / (bv1 * bv2)

    def __black_corr_intra_curves_factors(self
                                          , model_corr_mtx
                                          , curve_1: str
                                          , curve_2: str
                                          , tenor_1: datetime.date
                                          , tenor_2: datetime.date
                                          , factor_nb_1: int
                                          , factor_nb_2: int
                                          , opt_mat):
        """  Same as function above (__black_corr_intra_curves), but the factors are exposed

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
               self._beta_T(curve_1, tenor_1) * self._beta_T(curve_2, tenor_2) * \
               np.exp(- kv1[factor_nb_1] * self.__difference_to_market_date(tenor_1) - kv2[
                   factor_nb_2] * self.__difference_to_market_date(tenor_2)) * \
               (np.exp((kv1[factor_nb_1] + kv2[factor_nb_2]) * opt_mat) - 1.) / \
               (kv1[factor_nb_1] + kv2[factor_nb_2]) / (bv1 * bv2)

    # TODO: CHECK IF THIS METHOD IS EVEN NEEDED.
    def __black_corr_intra_curves_calib(self
                                        , curve_1: str
                                        , curve_2: str
                                        , solver='scipy_cobyla'):
        """ Calibrates the correlations between different curves.

        :param curve_1: curve 1 to for correlation calibration.
        :param curve_2: curve 2 for calibration.
        :param solver: solver to use in the OpenOpt
        """

        black_corr_intra_curve_vector = lambda model_corr_mtx, curve_1, curve_2, corr_len: \
            np.array([self.__black_corr_intra_curves(model_corr_mtx
                                                     , curve_1
                                                     , curve_2
                                                     , tenor
                                                     , tenor)
                      for tenor in range(corr_len)])

        black_corr_intra_curve_vector_optim = lambda model_corr_mtx, curve_1, curve_2, corr_len: \
            scipy.linalg.norm(black_corr_intra_curve_vector(model_corr_mtx, curve_1, curve_2, corr_len) -
                              self._market_corr[curve_1][curve_2])

        corr_len_real = len(self._market_corr(curve_1, curve_2))
        curve_1_nb_fact = self.nb_factors_for_asset(curve_1)
        curve_2_nb_fact = self.nb_factors_for_asset(curve_2)
        optim_pr = NSP(lambda corr_mtx_ravel: black_corr_intra_curve_vector_optim(
            corr_mtx_ravel.reshape((curve_1_nb_fact, curve_2_nb_fact)),
            curve_1, curve_2, corr_len_real),
                       self._factor_corr_mat(curve_1, curve_2).ravel(),
                       lb=self._factor_corr_mat(curve_1, curve_2, lb_ub_ind='lb').ravel(),
                       ub=self._factor_corr_mat(curve_1, curve_2, lb_ub_ind='ub').ravel()) \
            .solve(solver)

        correlation_matrix = np.array(optim_pr.xf).reshape((curve_1_nb_fact, curve_2_nb_fact))

        self.__factor_corr_mat_list[curve_1][curve_2] = correlation_matrix
        self.__factor_corr_mat_list[curve_2][curve_1] = correlation_matrix

        return self.__factor_corr_mat_list[curve_1][curve_2]

    def nb_factors_for_asset(self, asset : str) -> int:
        """ Number of factors per asset, placeholder perhaps for some other function.

        :param asset: asset for which the forward curve is obtained.
        """

        fwd_curve_names = [fwd_curve.fwd_name for fwd_curve in self.fwd_curves]
        if asset not in fwd_curve_names:
            raise ComSkewError('Requested curve {0} not in list of curves: {1}'.format( asset, fwd_curve_names))

        return 2

    def simulation_times( self
                        , sim_times : [np.ndarray, List[datetime.date], List[str], List[float] ]
                        , dcf = 365.25 ):
        """ Returns the simulation times from the list of sim_times.

        :param sim_times: new simulation times in one of the accepted formats
        :param dcf: day-count-factor
        """

        if isinstance(sim_times, np.ndarray):
            return sim_times, [self.mkt_date + datetime.timedelta(int(np.round(stf * dcf)))
                               for stf in sim_times]

        if (type(sim_times) == list) and (type(sim_times[0]) == datetime.datetime):
            sim_times_normalized = np.array([(st - self.mkt_date).days / 365. for st in sim_times])
            return sim_times_normalized, sim_times_normalized

        if (type(sim_times) == list) and (type(sim_times[0]) == float):
            sim_times_normalized = np.array(sim_times)
            return sim_times_normalized, [self.mkt_date + datetime.timedelta(int(np.round(stf * 365.))) for stf in sim_times_normalized]

    def DF( self
          , fwd_time : [float, datetime.date]
          , dcf = 365.25 ):
        """ Discount from self.mkt_date to fwd_time. Using basic discount curve.

        :param fwd_time: future time to discount to. can be '20140101', ...
        :param dcf: day-count factor.
        """

        if (type(fwd_time) is np.double) or (type(fwd_time) is float):
            time_diff = fwd_time

        elif isinstance(fwd_time, datetime.date):
            time_diff = self.__difference_to_market_date(fwd_time, dcf=dcf)

        else:
            raise ComSkewError('fwd_time given in function DF is not of form [float, datetime.date]')

        return self._discount_curve(time_diff)

    def __difference_to_market_date(self, fwd_date : datetime.date, dcf=365.25) -> float:
        """ Computes the difference to market date given the discount factor.
        """

        return (fwd_date - self.mkt_date).days / dcf

    def __fwd_square_vol (self
                          , asset       : str
                          , kappa_vec   : np.array
                          , sigma_vec   : np.array
                          , corr_matrix : np.array
                          , fwd_date_1: datetime.date
                          , fwd_date_2  : datetime.date ):
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

        # TODO: CHECK HOW TO INCORPORATE BETA_T BACK!!!
        #cross_1 = self._beta_T(asset, [fwd_date_2]) ** 2 * sigma_vec_row * corr_matrix * sigma_vec_col
        cross_1 = sigma_vec_row * corr_matrix * sigma_vec_col
        cross_2 = kappa_vec_row + kappa_vec_col

        time_to_fwd_date_1 = self.__difference_to_market_date(fwd_date_1)
        time_to_fwd_date_2 = self.__difference_to_market_date(fwd_date_2)

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
        beta  = self._beta_T(asset, [fwd_date])[0]  # number, not vector

        if kappa == 0.:
            return beta**2 * sigma**2 * (t_1 - t_0)

        return beta**2 * sigma**2 / (2. * kappa) * \
               np.exp(-2. * kappa * self.__difference_to_market_date(fwd_date)) * \
               (np.exp(2. * kappa * t_1) - np.exp(2. * kappa * t_0))

    def _V_cross_factor(self
                        , asset    : str
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

        kappa_1 = self._kappa_vec(asset)[factor_1]
        kappa_2 = self._kappa_vec(asset)[factor_2]
        kappa_12 = kappa_1 + kappa_2
        sigma_1 = self._sigma_vec(asset)[factor_1]
        sigma_2 = self._sigma_vec(asset)[factor_2]
        rho_12 = self._factor_corr_mat(asset, asset)[factor_1, factor_2]
        beta_1 = self._beta_T(asset, [fwd_date_1])[0]  # one forward date
        beta_2 = self._beta_T(asset, [fwd_date_2])[0]

        if kappa_12 == 0.:
            return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 * (t_1 - t_0)

        T_2 = 2.  # TODO: THIS IS BOGUS, DONT KNOW WHAT TO INSERT HERE

        return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 / kappa_12 * \
               (np.exp(-kappa_1 * (self.__difference_to_market_date(fwd_date_1) - t_1) - kappa_2 * (T_2-t_1)) -
                np.exp(-kappa_1 * (self.__difference_to_market_date(fwd_date_2) - t_0) - kappa_2 * (T_2-t_0)))

    def black_vol(self
                  , asset      : str
                  , fwd_date: datetime.date
                  , kappa_vec
                  , sigma_vec
                  , corr_matrix ):
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
                                     , self.mkt_date
                                     , fwd_date )

    def __option_tenor_for_fwd_tenor(self, asset: str, fwd_tenor : datetime.date) -> datetime.date:
        """ Returns the option tenor for a forward tenor.
            TODO: IMPROVE LATER.

        :param asset: asset we are requesting.
        :param fwd_tenor: a forward tenor for which the option tenor we are requesting.
        """

        return fwd_tenor

    def __distance_model_market_black_vol( self
                                         , asset
                                         , kappa_vec
                                         , sigma_vec
                                         , rho_vec
                                         , fwd_tenors = None ) -> float:
        """ Distance between model & market black volatility, used for calibration of the entire curve.
            Produces a sqaure sum of distance elements.

       :param asset: asset to consider, e.g. 'WTI'
       :param kappa_vec: kappa vector to calibrate
       :param sigma_vec: sigma vector to calibrate for asset asset
       :param rho_vec: correlation _vector_ to calibrate
       :param fwd_tenors: perhaps you want to restrict the forward tenors to a pre-defined set. If None,
                          fwd tenors from the forward curve are used.
        """

        # either use fwd_tenors provided, or use the fwd tenors from the fwd curve.
        fwd_tenors_used = fwd_tenors if fwd_tenors else self.fwd_curve_names(asset).fwd_tenors

        model_vol = [self.black_vol( asset
                                   , fwd_date
                                   , kappa_vec
                                   , sigma_vec
                                   , self._construct_corr_asset(asset, rho_vec))
                     for fwd_date in fwd_tenors_used]

        market_vol_curve = self.vol_curve_names(asset)
        market_vol = [market_vol_curve.atm_vol(fwd_date) for fwd_date in fwd_tenors_used]

        return sum([(market_vol_elt - model_vol_elt)**2
                    for market_vol_elt, model_vol_elt in zip(market_vol, model_vol)])

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

    def __factor_corr_mat(self, asset : str) -> np.ndarray:
        """ Returns the calibrated factor correlation matrix. e.g. 2x2 matrix for asset.

        :param asset: asset for which the correlation is returned.
        """

        # calibrated rho vector
        rho_vec = self._construct_corr_asset(asset, self.__kappa_sigma_rho(asset).xf[(2 * self.nb_factors_for_asset(asset)):])
        # transforming the rho_vec into the rho matrix
        rho_mat_len = np.sqrt(len(rho_vec))  # this is square matrix  TODO: THIS HERE IS WRONG!!!

        return rho_vec.reshape((rho_mat_len, rho_mat_len))

    # TODO: HERE SHOULD BE CHANGED TO ADD THIS CACHE
    # @lru_cache(maxsize=_BETA_T_CACHE_SIZE)
    def _beta_T(self
                , asset : str
                , tenors = None):
        """ Adjusts beta_T so that the atm vol is fitted perfectly.
            (assuming that kappa, sigma, rho has already been calibrated).
            The results are memoized.

        :param asset: name of the asset calibrated (e.g. 'WTI')
        :param tenors: list of forward tenors for which the beta is calibrated ( normally in form: List[datetime.date])
        """

        tenors_used = tenors if tenors else self.vol_curve_names(asset).tenors

        return np.array([self.vol_curve_names(asset).atm_vol(tenor) for tenor in tenors_used]) / \
               np.array([ self.black_vol( asset
                                        , tenor
                                        , self._kappa_vec(asset)
                                        , self._sigma_vec(asset)
                                        , self._factor_corr_mat(asset, asset) )
                         for tenor in tenors_used])

    def __numerical_dist_to_mktdate(self, fwd_date : datetime.date, dcf=365.25) -> float:
        """ Numerical distance to market date from fwd_date.
        """

        return (fwd_date - self.mkt_date).days / dcf

    def __deltas_to_strikes(self
                            , asset : str
                            , tenor_date     : datetime.date
                            , delta_vec_list : np.array ) -> np.array:
        """ Converts deltas to strikes for particular asset and tenor.

        :param asset: commodity asset to compute
        :param tenor_date: tenor considered.
        :returns: a vector of deltas from the strikes given in self.delta_vec_list
        """

        integrated_vol = self.vol_curve_names(asset).atm_vol(tenor_date) * \
                         np.sqrt(self.__difference_to_market_date(self.__option_tenor_for_fwd_tenor(asset, tenor_date)))

        return np.exp((scipy.stats.norm.ppf(delta_vec_list) - 0.5 * integrated_vol ) * integrated_vol) * \
               self.fwd_curve_names(asset).fwd_value(tenor_date)

    @staticmethod
    def __integr_analy( real_roots : np.ndarray
                      , A0         : float
                      , A1         : float
                      , A2         : float
                      , A3         : float
                      , A4         : float
                      , V          : float ):
        """ Integrate the polynomial between the roots, used for option calibration.
            Computes E[ p(x)_+ ]  where p(x) = A_0 x**4 + A_1 x**3 + A_2 x**2 + A_3 x + A_4

        :param real_roots: real roots of the polynomial described above.
        :returns: integrated value of the option w/ the prescribed parameters
        :rtype: float
        """

        Asigma = np.array([A0, A1, A2, A3, A4]) * np.array([1., V, V ** 2, V ** 3, V ** 4])  # A multiplied by sigmas
        nb_real_roots = len(real_roots)

        if nb_real_roots == 0:  # integrate polynomial function over whole of real axis
            if A4 > 0 or (A4 == 0 and A2 > 0) or (A4 == 0 and A2 == 0 and A0 > 0):
                return Asigma[0] + Asigma[2] + 3. * Asigma[4]

            return 0.

        # abbreviations
        cdf_below = ComMathsMixin._trunc_normal_below
        cdf_above = ComMathsMixin._trunc_normal_above
        cdf_interval = ComMathsMixin._trunc_normal_interval

        if nb_real_roots == 1:
            if A3 > 0:
                return np.sum(cdf_below(real_roots[0]) * Asigma)

            return np.sum(cdf_above(real_roots[0]) * Asigma)

        if nb_real_roots in [2, 3]:  # integrate over 2 intervals
            if A4 > 0:
                return np.sum(cdf_above(real_roots[0]) * Asigma) + \
                       np.sum(cdf_below(real_roots[1]) * Asigma)

            if A4 < 0.:
                return np.sum(cdf_interval(real_roots[0], real_roots[1]) * Asigma)

            if A4 == 0. and A3 != 0.:
                if A3 > 0.:
                    return np.sum(cdf_interval(real_roots[0], real_roots[1]) * Asigma) + \
                           np.sum(cdf_below(real_roots[2]) * Asigma)
                # A3 < 0
                return np.sum(cdf_above(real_roots[0]) * Asigma) + \
                       np.sum(cdf_interval(real_roots[1], real_roots[2]) * Asigma)
            if A4 == 0. and A3 == 0.:
                if A2 < 0.:
                    return np.sum(cdf_interval(real_roots[0], real_roots[1]) * Asigma)

                return np.sum(cdf_above(real_roots[0]) * Asigma) + \
                       np.sum(cdf_below(real_roots[1]) * Asigma)

        # elif nb_real_roots == 4:  # integrate over 3 intervals
        if A4 > 0:
            return np.sum(cdf_above(real_roots[0]) * Asigma) + \
                   np.sum(cdf_below(real_roots[3]) * Asigma) + \
                   np.sum(cdf_interval(real_roots[1], real_roots[2]) * Asigma)
        # A4 < 0
        return np.sum(cdf_interval(real_roots[0], real_roots[1]) * Asigma) + \
               np.sum(cdf_interval(real_roots[2], real_roots[3]) * Asigma)

    @staticmethod
    def __integr_num(A_V : np.array, call_put_ind : int, strike : float) -> float:
        """ Integrate numerically between the roots of the polynomials.
             IMPORTANT: For testing purposes only, in prod. use the code in integr_analy.

        :param A_V: array of A0, A1, A2, A3, A4, V
        :param call_put_ind: indicator for call (1) or put (-1)
        :param strike: option strike.
        :returns: integrated option value for TODO: COMPLETE HERE.
        """

        from scipy.integrate import quad

        A0, A1, A2, A3, A4, V = A_V
        if call_put_ind == 1:
            A0 -= strike
        else:  # put
            A0 = strike - A0
            A1 = - A1
            A2 = - A2
            A3 = - A3
            A4 = - A4

        return quad(lambda x: np.max([A0 + A1 * V * x + A2 * V**2 * x**2 +
                                      A3 * V**3 * x**3 + A4 * V**4 * x**4, 0.]) / \
                                      np.sqrt(2. * np.pi) * np.exp(- x**2 / 2.)
                   , -np.inf
                   , np.inf)[0]

    def _polynomial_european(self
                             , asset_nb : str
                             , C_vec
                             , opt_mat_idx
                             , strike       : float
                             , call_put_ind : int
                             , ttm          : float
                             , debug_mode = False) -> float:
        """
        value of european call option in skew model with strike
        call_put_ind ... 1 for call, -1 for put

        :param asset_nb: asset to consider, e.g. 'WTI'.
        :param C_vec: vector of skew parameters
        :param opt_mat_idx:
        :param strike: strike of the option TODO
        :param call_put_ind: indicator whether this is a call (1) or a put (-1)
        :param ttm: time to maturity of the option.
        :param debug_mode: whether to run in debug mode
        """

        # obtaining the coefficients
        A0, A1, A2, A3, A4, V = self._skew_params(asset_nb, C_vec, opt_mat_idx)
        if call_put_ind == 1:
            A0 -= strike
        else:  # put
            A0 = strike - A0
            A1 = - A1
            A2 = - A2
            A3 = - A3
            A4 = - A4

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

        print('PARAMS {0}'.format([A0, A1, A2, A3, A4, V]))

        return self.DF(ttm) * self.__class__.__integr_analy( real_roots / V, A0, A1, A2, A3, A4, V )

    def _default_deltas_for_skew(self):
        """ Default deltas to be used for calibration. - simple here, can be overwritten
        """

        return np.linspace(0.1, 0.9, 10)

    def __model_vol_surface(self, asset : str, C_vec, fwd_date : datetime.date, deltas = None):
        """ Computes model vols for asset, C_vec, fwd_idx.

        :param asset: commodity considered.
        :param C_vec: skew vector
        :param deltas: deltas to be used for calibration.
        """

        deltas_used = deltas if deltas else self._default_deltas_for_skew()

        strikes = self.__deltas_to_strikes(asset, fwd_date, deltas_used)
        fwd_value = self.fwd_curve_names(asset).fwd_value(fwd_date)
        cp_ind = np.array([1 if strike >= fwd_value else -1 for strike in strikes])

        # TODO: check if self.__difference_to ... is the correct parameter.
        option_tenor = self.__option_tenor_for_fwd_tenor(asset, fwd_date)  # option tenor corresponding to fwd_date
        option_prices = [self._polynomial_european( asset
                                                  , C_vec
                                                  , fwd_date
                                                  , strike
                                                  , cp
                                                  , self.__difference_to_market_date(option_tenor))
                         for strike, cp in zip(strikes, cp_ind)]
        print("OPTION P: {0}".format(option_prices))
        df = self.DF(option_tenor)

        return np.array([black_vol_inverse( fwd_value
                                          , strike
                                          , opt_price
                                          , self.__difference_to_market_date(option_tenor)
                                          , df
                                          , cp
                                          , self.black_vol_inverse_tol)
                         for opt_price, strike, cp in zip( option_prices, strikes, cp_ind ) ] )

    def _opt_fct_skew( self
                     , asset     : str
                     , fwd_dates : List[datetime.date]
                     , deltas = None  ):
        """ Optimization function to minimize over the fwd_dates.

        :param asset: asset for which the skew function is calibrates, e.g. ('WTI')
        :param fwd_dates: list of forward dates for calibration of the skew.
        :param deltas: deltas for which to calibrate the skew
        """

        # penalize the calibrated funtion for values of C where positive forward prices.
        # penalization level is 10000

        deltas_used = deltas if deltas else self._default_deltas_for_skew()

        C_calib = {}
        for fwd_date in fwd_dates:
            diff_to_mkt_date = self.__difference_to_market_date(fwd_date)
            fwd_value = self.fwd_curve_names(asset).fwd_value(fwd_date)
            implied_vols = np.array([self.vol_curve_names(asset).implied_vol( fwd_date
                                                                            , np.exp(delta) * fwd_value
                                                                            , diff_to_mkt_date)
                                     for delta in deltas_used ])

            C_calib[fwd_date] = NLP( lambda C_vec: scipy.linalg.norm(self.__model_vol_surface(asset, C_vec, fwd_date) - implied_vols)
                                   , np.array([1., 0., 0.])
                                   , iprint = -1 )\
                                   .solve(self.__class__.NLP_SOLVER).xf

        return C_calib

    def __c_vec_calibrate( self
                         , asset     : str
                         , fwd_dates : List[datetime.date]
                         , multi_thread_ind = False ):
        """ Returns the skew parameters for asset. If done the first time, it calibrates the parameters,
            otherwise it returns the stored value.

        :param asset: the asset to calibrate, such as 'wti'
        :param fwd_dates: forward dates for which to calibrate
        :param multi_thread_ind: indicator whether to use multiple threads
        """

        if not multi_thread_ind:
            return np.array(self._opt_fct_skew(asset, fwd_dates))

        # using multithreading
        with Pool(processes=cpu_count()) as pool:
            curr_nb_tenors = len(fwd_dates)
            C = pool.map(opt_fct_skew_wrap, zip([self] * curr_nb_tenors, [asset] * curr_nb_tenors, fwd_dates))

            return np.array(C)

    def _complete_corr_mtx(self, nb_steps=300) -> np.ndarray:
        """  Generates the factor correlation matrix from a list of list of corr. matrices
             gathered in __factor_corr_mat_list.

        :param nb_steps: number of steps to converge to the correlation matrix, maybe 30 steps is enough
        :returns: None, just sets the self.__completeCorrMat
        """

        if not self.__regenerate_complete_corr_mtx:
            return self.__complete_corr_mtx

        # starting to generate the complete correlation matrix

        total_nb_factors = sum([self.nb_factors_for_asset(fwd_curve.fwd_name) for fwd_curve  in self.fwd_curves])
        self.__complete_corr_mtx = np.zeros((total_nb_factors, total_nb_factors))

        # sets the large correlation building blocks
        for asset_1 in self.fwd_curves:
            for asset_2 in self.fwd_curves:
                # TODO: HERE COMPLETELY WRONG
                self.__complete_corr_mtx[self.__factor_positions(asset_1.fwd_name), self.__factor_positions(asset_2.fwd_name)] = self._ # self.__factor_corr_mtx(asset_1, asset_2)

        # find the closest matrix that is positive semidefinite
        self.__complete_corr_mtx = near_corr_simple(self.__complete_corr_mtx, nb_steps)
        while not (np.linalg.eig(self.__complete_corr_mtx)[0] > 0.).all():
            d1, v1 = np.linalg.eig(self.__complete_corr_mtx)
            d1p = np.diag(np.maximum(d1, 1.e-16))
            self.__complete_corr_mtx = np.dot(v1, np.dot(d1p, v1.transpose()))

        return self.__complete_corr_mtx

    def _corr_mtx_for_assets(self, assets : List[str], tenors_for_assets: List[List[datetime.date]]) -> np.ndarray:
        """ Returns the correlation matrix for the assets in a list.

        :param assets: list of assets, e.g. ['WTI', 'ERCOT_PEAK']
        :param tenors_for_assets: tenors for each individual asset, e.g. [[datetime.date(2016, 1, 1), datetime.date(2016, 2,1)], [datetime.date(2016, 3, 1), datetime.date(2016, 4, 1)]]
        """

        mtx_size = sum([len([x ] )])

        corr_mtx = np.empty((mtx_size, mtx_size))  # resulting corrlation matrix

        for x in X:
            idx_1 = 1
            idx_2 = 2  # TODO: FINISH THIS index to factor ???
            x[idx_1, idx_2] = self.__black_corr_intra_curves( model_corr_mtx
                                                            , curve_1
                                                            , curve_2
                                                            , tenor_1
                                                            , tenor_2)

        return corr_mtx

    def _var_covar_mtx( self
                      , asset_nb : str
                      , fwd_idx
                      , i  : int
                      , j  : int
                      , t_idx
                      , sim_times ):
        """ Generate covar mtx, part of LN simulation, used in simulate_curves.

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
                             , nb_simulations : int ) -> np.ndarray:
        """ Simulates the standard normal random variables with specified correlations, and returns the
        matrix of size (nb_factors x nb_correlations)  # TODO: CHECK IF THIS IS NOT REVERSED.

        :param corr_mtx: correlation matrix, a nb_factors x nb_factors matrix.
        :param nb_simulations: number of simulations from the factors.
        """

        return np.random.multivariate_normal( np.zeros(nb_factors)
                                            , corr_mtx
                                            , size = nb_simulations )

    def simulate_curves( self
                        , assets           : List[str]
                        , nb_simulations   : int
                        , simulation_times : List[datetime.date]
                        , tenor_list       : List[datetime.date]
                        , set_seed         = None) -> np.array :
        """ Simulate all curves for desired simulation times.

        Generates a 3-dimensional array
        0-th dimension: asset
        1-st dimension: simulation times
        2-nd dimension: curve
        3-rd dimension: repeats of the curve

        :param assets: list of assets for which to simulate
        :param nb_simulations: self. explanatory
        :param simulation_times: simulation times for the forwards.
        :param tenor_list: list of tenors which to simulate
        :param set_seed: seed, if needed, can be left to None
        :returns: a matrix of simulated paths
        """

        np.random.seed(set_seed)
        simulated_curves = {}
        fwd_c_col = {}

        for com_curve in assets:
            com_fwd_curve = self.fwd_curve_names(com_curve)
            simulated_curves[com_curve] = np.empty((len(simulation_times), len(tenor_list), nb_simulations))  #  if not cuda_ind else gpa.zeros(sim_curves_shape, dtype=rn_type)
            fwd_c_col[com_curve] = com_fwd_curve.fwd_value(tenor_list)
            simulated_curves[com_curve][0, :, :] = np.array(fwd_c_col[com_curve]).reshape((len(tenor_list), 1))

        # X and X_prev are simulated factors
        X = {}
        X_prev = {}
        # for com_curve in assets:  # self.fwd_curves:
        for asset in assets:
            X_mat_shape = (len(fwd_c_col[asset]), nb_simulations)
            X[asset] = np.zeros(X_mat_shape)
            X_prev[asset] = np.empty(X_mat_shape)

        # looping over time steps
        #   simulates ln process, basis for skew as well
        #   t_i ... idx of sim_time
        #   fact_sum ... factors of the individual assets
        sim_times_numeric = [self.__difference_to_market_date(t_datetime) for t_datetime in simulation_times]
        for t_i, t_value in enumerate(sim_times_numeric):

            nb_factors_by_asset = [self.nb_factors_for_asset(asset) for asset in assets]
            total_nb_factors = sum(nb_factors_by_asset)

            factor_corr_mat = self.__factor_corr_mat_multiple(assets)
            simulated_rn    = self.__class__.__simulate_std_normal( total_nb_factors, factor_corr_mat, nb_simulations )

            sims_Z_unit = np.dot( np.linalg.inv(np.linalg.cholesky(factor_corr_mat))
                                , simulated_rn.transpose())

            for asset in assets:
                for tenor_idx, tenor_nb in enumerate(tenor_list):
                    # prepare cov mtx
                    nb_factors_asset = self.nb_factors_for_asset(asset)
                    cov_chol = np.linalg.cholesky(np.array([[self._var_covar_mtx(asset, tenor_nb, i, j, t_i, sim_times_numeric)
                                                             for j in range(nb_factors_asset)]
                                                            for i in range(nb_factors_asset)]))
                    delta_X = np.sum(np.dot(cov_chol, sims_Z_unit), axis=0)
                    # quadratic variation of delta_X, also q_v = V_u
                    qv = np.sum([[self._V_cross_factor( asset
                                                      , factor_1
                                                      , factor_2
                                                      , tenor_nb
                                                      , tenor_nb
                                                      , 0. if t_i == 0 else sim_times_numeric[t_i - 1]
                                                      , t_value )
                                  for factor_1 in range(nb_factors_asset)]
                                 for factor_2 in range(nb_factors_asset)])

                    # replacing old w/ new and generating new
                    X_prev[asset][tenor_idx, :] = X[asset][tenor_idx, :]
                    X[asset][tenor_idx, :] = X_prev[asset][tenor_idx, :] + delta_X

                    # F_res = F_u * (1. + X_u + 0.5 * c1 * (X_u**2 - V_u) +
                    #                c2 * (X_u**3 - 3. * X_u * V_u) / 6. +
                    #                c3 * (X_u**4 - 6. * V_u * X_u**2 + 3. * V_u**2) / 24.)
                    # self.simulated_curves[asset][t_i, tenor_idx, :] = F_res
                    c1, c2, c3 =  self._c_vec(asset, tenor_nb)
                    print('C0, C1, C2: {0}, {1}, {2}'.format(c1, c2, c3))
                    skew_fom( self.fwd_curve_names(asset).fwd_value(tenor_nb)
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
        cums = np.cumsum(self.nb_factors_for_asset(asset))
        fact_sum = np.zeros(len(cums)+1, dtype=np.int)
        fact_sum[1:(len(cums)+1)] = cums

        return slice(fact_sum[asset_nb], fact_sum[asset_nb+1])

    def simulate_curves_fom(self
                            , asset          : str
                            , nb_simulations : int
                            , sim_times      : List[datetime.date]
                            , tenors_list = None
                            , seed        = None
                            , model       = 'skew'):
        """ Simulate first of month (fom) curves.

        generates a list of 2 dim arrays:
           1-st dim: tenor
           2-nd dim: simulation

        :param asset: asset for which to simulate, e.g. 'WTI'
        :param nb_simulations: number of simulations
        :param sim_times: simulation times
        :param tenors_list: list of tenors to simulate TODO: LOTS TO DISCUSS HERE!!!
        :param model: which model: 'ln' or 'skew', default 'skew'
        """

        np.random.seed(seed=seed)

        sim_times = self.vol_curve_names(asset).vol_tenors if not tenors_list else self.vol_curve_names(asset).vol_tenors[tenors_list]
        complete_corr_mtx = self._complete_corr_mtx()

        # looping over tenors
        #    t_i ... idx of sim_time (also tenor)
        #    fact_sum ... factors of the individual assets
        sim_fom = np.empty((len(tenors_list), nb_simulations))  # nb_tenors x nb_simulations
        for tenor_idx, tenor_date in enumerate(tenors_list):
            t_curr = sim_times[tenor_idx]
            F_curr = self.fwd_curve_names(asset)[tenor_date]
            nb_factors = np.sum(self.nb_factors_for_asset(asset))  # total number of factors
            simulated_rn = np.random.multivariate_normal( np.zeros(nb_factors)
                                                        , complete_corr_mtx
                                                        , size=nb_simulations)
            nb_factors_asset = self.nb_factors_for_asset[asset]
            new_cov_mat = np.array([[self._var_covar_mtx(self, asset, tenor_date, i, j, tenor_idx, sim_times)
                                    for j in range(nb_factors_asset)]
                                    for i in range(nb_factors_asset)])

            new_chol = np.linalg.cholesky(new_cov_mat)
            old_cov_mat = complete_corr_mtx[self.__factor_positions(asset), self.__factor_positions(asset)]
            old_chol = np.linalg.cholesky(old_cov_mat)

            sims_Z = simulated_rn[:, self.__factor_positions(asset)].transpose()
            sims_Z_unit = np.dot(np.linalg.inv(old_chol), sims_Z)
            delta_X = np.sum(np.dot(new_chol, sims_Z_unit), axis=0)
            qv = np.sum([[self._V_cross_factor(asset, factor_1, factor_2, tenor_date, tenor_date, 0., t_curr)
                          for factor_1 in range(nb_factors_asset)]
                         for factor_2 in range(nb_factors_asset)])

            if model == 'ln':
                sim_fom[tenor_idx, :] = F_curr * np.exp(delta_X - 0.5 * qv)
            else:  # skew model
                c0, c1, c2 = self._c_vec(asset, tenor_date)
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

    def _skew_params(self
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
        f0t = self.fwd_curve_names(asset).fwd_value(fwd_date)

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
