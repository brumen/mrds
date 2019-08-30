#
#   skew model for forward curves
#

import datetime
import numpy as np
import scipy
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats
import scipy.optimize
import scipy.interpolate  # spline package
from openopt import NLP
import multiprocessing as mp

from functools import lru_cache

from typing import List, Dict

from mrds_maths    import ComMathsMixin
from mrds_defaults import ComSkewDefaultsMixin

import matplotlib as mpl
mpl.use('TkAgg')

# mrds local imports
import ds
import vols.vols
from vols.vols import getVolObject
import near_corr
import correlations as corrs

from forward_curve import FwdCurve
from vols.vols     import Volatility

import quartic.quartic_cy as quartic_cy

import opd.opd_avx as opd_avx

import logging


logger = logging.Logger(__name__)


class ComSkewError(Exception):
    """ Base class for ComSkew model exceptions.
    """

    pass


def opt_fct_skew_wrap(arg, **kwarg):
    """ Wrapper for the skew MRD model calibration function.
    """

    return ComSkew.__opt_fct_skew(*arg, **kwarg)


class ComSkew(ComMathsMixin, ComSkewDefaultsMixin):
    """ Base class of the commodity skew market model.
    """

    NLP_SOLVER           = 'scipy_cobyla'
    _LRU_CACHE_SIZE_CALIB = 5  # lru for number of factors.

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
            return corrs.corr_hyp_sec_mat(same_asset_corr, range(self.nb_factors_for_asset(asset_1)))

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
    def nb_assets(self):
        return len(self._com_fwd_curves)

    @property
    def fwd_curves(self):
        """ Curve names for the commodity curves in the model.
        """

        return self._com_fwd_curves

    @property
    def vol_curves(self):
        return self._com_vol_curves


# TODO: INCLUDE THIS HERE
#        , 'corr_init': np.array([[1., 0.5], [0.5, 1.]])
#        , 'corr_lb': np.array([[1., -0.99], [-0.99, 1.]])
#        , 'corr_ub': np.array([[1., 0.99], [0.99, 1.]])} ):


    def __init__( self
                , mkt_date      : datetime.date
                , fwd_curves    : Dict[FwdCurve]
                , volCurves  # : List[Volatility]
                , calcDate     = None):

        """ Initialization of the skew model.

        :param mkt_date: market date
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param comVolCurves: commodity vol curves, in case they are different than forward curves.
        """

        self._mktDate       = mkt_date
        self.__mktDateChange= True  # indicator whether the market date has changed and everything needs to be recalculated.
        self._calcDate      = calcDate if calcDate else mkt_date
        self._com_fwd_curves  = fwd_curves
        self._com_vol_curves  = volCurves
        nb_assets           = len(self._com_fwd_curves)

        # initial parameters
        self.__sigmaInit = None
        self.__sigmaLB   = None
        self.__sigmaUB   = None
        self.__kappaInit = None
        self.__kappaLB   = None
        self.__kappaUB   = None

        # skew parameters dictionary, one np.array of three elements for each forward curve and each tenor
        self._CVecList   = {}
        self.__betaTList = {}

        # initial value of the calibrated params
        for fwdCurveName, fwdCurveObj in self._com_fwd_curves:  # iterate through curves

            # potential calibration parameters
            self._CVecList[fwdCurveName] = ComSkew._oneZeroZeroMatrix(len(self._com_fwd_curves[fwdCurveName]), 3)

            # initial value of the sigmaInit
            # TODO: USE THE INIT VERSION OF SIGMA
            self.__betaTList[fwdCurveName] = np.ones(len(self.forward_tenors_list[fwdCurveName]) )


        # indicator functions - whether the values are updated
        # indicator function for the sigma, kappa calibration
        self.sigma_kappa_calib_indicator_list = np.repeat(False, nb_assets)
        self.skew_calib_indicator_list = np.repeat(False, nb_assets)
        self.sim_date_indicator = False
        self.days_nb_const_ind = False  # monthly dat numbers are constr.
        self.simulate_spot_rn_ind = False  # indic. for random number for spot sim.
        self._cash_corr = np.eye(self.nb_assets)

        # simulated curves section, placeholders
        self.sim_rv_ind = { 'ind'   : False
                          , 'nb_sim': 0 }
        self.sim_rv = None

        # discount function
        self.__discount_function = None

        self.__blackVolInverseTol = 1e-4  # default value of the black vol inverse parameter

        # stored functions to be used by cuda if necessary
        self.__FskewFctCuda = {}  # empty dict, no skew functions stored.

    def market_corr_list(self, asset_1, asset_2):
        return None

    def model_corr_list(self, asset_1, asset_2):
        return None

    def overwrite_market_corr(self, asset_1, asset_2, overwr):
        """
        Overwrites read corr. with manual

        """

        logger.info('Market correlation vector overwritten with: ' + overwr)
        self.market_corr_list[asset_1][asset_2] = overwr


    @property
    def blackVolInverseTol(self):
        return self.__blackVolInverseTol

    @blackVolInverseTol.setter
    def blackVolInverseTol(self, newInverseTol):
        self.__blackVolInverseTol = newInverseTol

    @classmethod
    def fromDb( cls
              , mktDate: datetime.date
              , comFwdCurveNames ):
        """
        Constructs the class by reading forward and vol curves from the database.

        """

        fwdCurves = [FwdCurve.fromDB(mktDate, fwdCurveName) for fwdCurveName in comFwdCurveNames]
        volCurves = [getVolObject(fwdCurveName, mktDate)    for fwdCurveName in comFwdCurveNames]

        return cls(mktDate, fwdCurves, volCurves)

    @property
    def mkt_date(self) -> datetime.date:
        return self._mktDate

    @mkt_date.setter
    def mkt_date(self, newMarketDate : datetime.date):
        """
        Sets the new market date, updates all the curves accordingly.

        """

        self._mktDate = newMarketDate
        self.__mktDateChange = True
# TODO: CHECK THESE
#        for asset_ch in range(self.nb_assets):
#            self.update_one_asset(newMarketDate, asset_ch)

    @property
    def calc_date(self) -> datetime.date:
        return self._calcDate

    @calc_date.setter
    def calc_date(self, new_calc_date : datetime.date):
        """ Sets the new calculation date, updates all the curves accordingly.

        :param new_calc_date: new caluclation date.
        """

        self._calcDate = new_calc_date

    def nb_factors_for_asset(self, asset):
        """ Number of factors per asset, placeholder perhaps for some other function.
        """

        return 2

    # TODO: NEXT THREE FUNCTIONS ARE NOT DEFINED!!!
    def __factorCorrMat(self, asset1, asset2):
        """
        Factor correlation matrix between asset1 and asset2.

        :param asset1: first asset.
        :param asset2: second asset.
        """

        return self.__default_factor_corr_mat_fct(asset1, asset2)

    def __factorCorrMatLB(self, asset_1, asset_2):
        """
        Lower bound on the factor correlation matrix.

        """

        return self.__default_factor_corr_mat_fct_lb_ub(asset_1, asset_2, lb_ub_ind='lb')

    def __factorCorrMatUB(self, asset_1, asset_2):
        """
        Upper bound on the factor correlation matrix.
        """

        return self.__default_factor_corr_mat_fct_lb_ub(asset_1, asset_2, lb_ub_ind='ub')

    def simulation_times( self
                        , sim_times : [np.ndarray, List[datetime.date], List[str], List[float] ] ):
        """ Update simulation times for the model.

        :param sim_times: new simulation times in one of the accepted formats
        """

        if type(sim_times) == np.ndarray:
            return sim_times, [self.mkt_date + datetime.timedelta(int(np.round(stf * 365.)))
                               for stf in sim_times]

        if (type(sim_times) == list) and (type(sim_times[0]) == datetime.datetime):
            sim_times_normalized = np.array([(st - self.mkt_date).days / 365. for st in sim_times])
            return sim_times_normalized, sim_times_normalized

        if (type(sim_times) == list) and (type(sim_times[0]) == str):
            return np.array([(ds.convert_str_datetime(date_) - self.mkt_date).days / 365.
                                               for date_ in sim_times]), \
                    sim_times

        if (type(sim_times) == list) and (type(sim_times[0]) == float):
            sim_times_normalized = np.array(sim_times)
            return sim_times_normalized, [self.mkt_date + datetime.timedelta(int(np.round(stf * 365.))) for stf in sim_times_normalized]

    def updateMarketDateOneAsset(self, newMarketDate : datetime.date) -> None:
        """
        Updates the date to the new market date, and updates the curves and vols accordingly.

        :param newMarketDate: new date that one wants to set.
        """

        for comCurves in self._com_fwd_curves:
            comCurves._mktDate = newMarketDate  # TODO: _mktDate SHOULD BE DIFFERENT IN THE FwdCurve class

        for volCurve in self._com_vol_curves:
            volCurve.mkt_date = newMarketDate

    @property
    def _discount_function(self):
        """ Returns the discount function for the market date.
        """

        if self.__discount_function:
            return self.__discount_function

        self.__discount_function = ds.read_discount_curve(self.mkt_date)
        return self.__discount_function

    def DF( self
          , fwd_time : [str, np.double, float]
          , dcf = 365.25 ):
        """ Discount from self.mkt_date to t

        :param fwd_time: future time to discount to. can be '20140101', ...
        :param dcf: day-count factor.
        """

        if (type(fwd_time) is np.double) or (type(fwd_time) is float):
            time_diff = fwd_time

        elif isinstance(fwd_time, str):
            t_dt = ds.convert_str_datetime(fwd_time)
            time_diff = self.__difference_to_market_date(t_dt, dcf=dcf)

        elif isinstance(fwd_time, datetime.datetime):
            time_diff = self.__difference_to_market_date(fwd_time, dcf=dcf)

        return scipy.interpolate.splev(time_diff, self._discount_function)


    @staticmethod
    def _construct_corr(mtx_size, theta_vector):
        """ Constructs and upper triangular matrix from a vector theta_vector, first row is from the rho matrix.

        :param mtx_size: matrix size.
        :param theta_vector:
        """

        utm = np.triu(np.ones((mtx_size, mtx_size))) # upper triangular matrix
        utm_diag_ones = np.diag(np.diag(utm))
        utm = utm - utm_diag_ones
        utm[utm==1] = theta_vector
        utm += utm.transpose() + utm_diag_ones

        return utm

    def _construct_corr_asset (self, asset_nb, theta_vector):
        return ComSkew._construct_corr(self.nb_factors_for_asset(asset_nb), theta_vector)

    def __difference_to_market_date(self, fwdDate : datetime.date, dcf=365.25) -> float:
        """ Computes the difference to market date given the discount factor.
        """

        return (fwdDate - self.mkt_date).days / dcf

    def __fwd_square_vol ( self
                         , asset       : str
                         , kappa_vec   : np.array
                         , sigma_vec   : np.array
                         , corr_matrix : np.array
                         , fwdDate     : datetime.date
                         , t           : float ):
        """ Computes forward integrated square vol (function V) from t to fwd date fwd_date TODO: CHECK THIS DESCRIPTION

        :param asset: commodity to be considered. (e.g. 'WTI')
        :param kappa_vec: vec of kappas
        :param t:
        """

        nbFactors = self.nb_factors_for_asset(asset)

        sigma_vec_row = sigma_vec.reshape((1, nbFactors))
        sigma_vec_col = sigma_vec.reshape((nbFactors, 1))
        kappa_vec_row = kappa_vec.reshape((1, nbFactors))
        kappa_vec_col = kappa_vec.reshape((nbFactors, 1))

        cross_1 = self._betaT(asset, [fwdDate]) ** 2 * sigma_vec_row * corr_matrix * sigma_vec_col
        cross_2 = kappa_vec_row + kappa_vec_col

        time_to_fwd_date = self.__difference_to_market_date(fwdDate)

        return np.sum(cross_1 * (np.exp(-cross_2 * ( time_to_fwd_date - t)) -
                                 np.exp(-cross_2 * time_to_fwd_date)) / cross_2)

    def __VOneFactor(self
                    , asset      : str
                     , factor_nb  : int
                     , fwd_date   : datetime.date
                     , t_0
                     , t_1):
        """
        Computes integrated volatility V only for one factor (factor_nb)

        """

        kappa = self._kappa_vec(asset)[factor_nb]  # TODO: THESE TWO LINES CORRECT
        sigma = self._sigma_vec(asset)[factor_nb]  # TODO: THESE TWO LINES CORRECT
        beta  = self._betaT(asset, fwd_date)
        T     = self.__difference_to_market_date(fwd_date)

        if kappa == 0.:
            return beta**2 * sigma**2 * (t_1 - t_0)

        return beta**2 * sigma**2 / (2. * kappa) * \
               np.exp(-2. * kappa * T) * \
               (np.exp(2. * kappa * t_1) - np.exp(2. * kappa * t_0))

    def _V_cross_factor(self
                        , asset_nb : str
                        , factor_1 : int
                        , factor_2 : int
                        , fwd_date_1 : datetime.date
                        , fwd_date_2 : datetime.date
                        , t_0      : float
                        , t_1      : float):
        """ Computes cross integrated vol. V for only one factor.

        t0, t1 ... vol is copmputed from t_0 to t_1
        """

        kappa_1 = self._kappa_vec(asset_nb)[factor_1]
        kappa_2 = self._kappa_vec(asset_nb)[factor_2]
        kappa_12 = kappa_1 + kappa_2
        sigma_1 = self._sigma_vec(asset_nb)[factor_1]
        sigma_2 = self._sigma_vec(asset_nb)[factor_2]
        rho_12 = self.__factorCorrMat(asset_nb, asset_nb)[factor_1, factor_2]
        beta_1 = self._betaT(asset_nb, fwd_date_1)
        beta_2 = self._betaT(asset_nb, fwd_date_2)
        T_1    = self.__difference_to_market_date(fwd_date_1)
        T_2    = self.__difference_to_market_date(fwd_date_2)

        if kappa_12 == 0.:
            return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 * (t_1 - t_0)

        return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 / kappa_12 * \
               (np.exp(-kappa_1 * (T_1 - t_1) - kappa_2 * (T_2-t_1)) -
                np.exp(-kappa_1 * (T_1-t_0) - kappa_2 * (T_2-t_0)))

    # def __fwd_total_sqr_vol(self, asset_nb, fwd, t_0, t_1):
    #     """ Total integrated volatility between t_0 and t_1
    #
    #
    #     """
    #     v_total_1 = np.sum([self.__VOneFactor(asset_nb, factor_nb, fwd, t_0, t_1)
    #                         for factor_nb in range(self.nb_factors_for_asset(asset_nb))])
    #     v_total_2 = np.sum(np.sum([[self._V_cross_factor(asset_nb,
    #                                                      factor_1, factor_2,
    #                                                      fwd, fwd, t_0, t_1)
    #                                 for factor_2 in range(factor_1 + 1, self.nb_factors_for_asset(asset_nb))]
    #                                for factor_1 in range(self.nb_factors_for_asset(asset_nb))]))
    #
    #     return v_total_1 + v_total_2

    def black_vol( self
                 , asset      : str
                 , kappa_vec
                 , sigma_vec
                 , corr_matrix
                 , fwdDate    : datetime.date ):
        """ Computes the model black vol.
        TODO: THIS FUNCTION IS NEARLY USELESS !!!

        """

        return np.sqrt(self.__fwd_square_vol(asset
                                             , kappa_vec
                                             , sigma_vec
                                             , corr_matrix
                                             , fwdDate  # TODO: THE NEXT ARGUMENT IS WRONG
                                             , fwdDate) / self.__difference_to_market_date(fwdDate))

    def black_vol_current(self, asset_nb : str, fwd_date : datetime.date) -> float:
        """ Computes black vol until option maturity for the given model parameters.

        :param asset_nb: asset number (e.g. 'WTI')
        :param fwd_date: forward date
        :returns: black volatility for the model TODO: REWRITE THESE DESCRIPTIONS
        """

        return self.black_vol(asset_nb
                              , self._kappa_vec(asset_nb)
                              , self._sigma_vec(asset_nb)
                              , self.__factorCorrMat(asset_nb,asset_nb)
                              , fwd_date)

    def V_fct_current(self, asset_nb, fwd_idx, t):
        return self.__fwd_square_vol(asset_nb
                                     , self._kappa_vec(asset_nb)
                                     , self._sigma_vec(asset_nb)
                                     , self.__factorCorrMat(asset_nb, asset_nb)
                                     , fwd_idx
                                     , t)

    def optionTenorForFwdTenor(self, asset: str, fwdTenor : datetime.date ) -> datetime.date :
        """ Returns the option tenor for a forward tenor.

        :param asset: asset we are requesting.
        :param fwdTenor: a forward tenor for which the option tenor we are requesting.
        """

        # TODO: THIS NEEDS TO BE IMPORTED !!!
        return fwdTenor  # TODO: FOR NOW, LATER IMPROVE

    def __modelBlackVol( self
                       , asset_nb
                       , kappa_vec
                       , sigma_vec
                       , rho_vec ):

        return np.sum((np.array([self.black_vol(asset_nb, kappa_vec, sigma_vec,
                                             self._construct_corr_asset(asset_nb, rho_vec), T)
                                 for T in range(self.nb_assets[asset_nb])])  # TODO: THIS nb_assets is WRONG
                       - self.__volObjectList[asset_nb].atmVol()) ** 2)  # TODO: for all forwards

    @lru_cache(maxsize=10)  # TODO: FIX THIS HERE
    def black_vol_calibration( self
                             , asset_nb : str ):
        """
        Calibrates kappa and sigma and rho parameters of the log-normal part of the model.

        :param asset_nb: asset to be calibrated
        """

        nbf = self.nb_factors_for_asset(asset_nb)

        # extracting the upper triangular part of the correlation matrix 
        fcm_init = self.__factorCorrMat(asset_nb, asset_nb)
        fcm_lb   = self.__factorCorrMatLB(asset_nb, asset_nb)
        fcm_ub   = self.__factorCorrMatUB(asset_nb, asset_nb)

        # optimization run
        optim_res = NLP( lambda kappa_sigma_rho_vec: self.__modelBlackVol(kappa_sigma_rho_vec[0:nbf],
                                                                          kappa_sigma_rho_vec[nbf:(2*nbf)],
                                                                          kappa_sigma_rho_vec[(2*nbf):])
                       , np.concatenate([ self.kappaDefault(nbf, 'init')
                                        , self.sigmaDefault(nbf, 'init')
                                        , np.triu(fcm_init, 1)[np.triu(fcm_init, 1) != 0] ])
                       , lb = np.concatenate([self.kappaDefault(nbf, 'lb'),
                                              self.sigmaDefault(nbf, 'lb'),
                                              np.triu(fcm_lb, 1)[np.triu(fcm_lb, 1) != 0] ])
                       , ub = np.concatenate([self.kappaDefault(nbf, 'ub'),
                                              self.sigmaDefualt(nbf, 'ub'),
                                              np.triu(fcm_ub, 1)[np.triu(fcm_ub, 1) != 0] ])
                       , iprint = -1 )\
                       .solve(ComSkew.NLP_SOLVER)

        return optim_res

    @lru_cache(maxsize=_LRU_CACHE_SIZE_CALIB)
    def _kappa_vec(self, asset : str) -> np.ndarray:
        """ Holds the kappa vector for a particular asset.

        :param asset: asset to compute the kappa vector of.
        """

        return self.black_vol_calibration(asset).xf[0:self.nb_factors_for_asset(asset)]  # number of factors

    @lru_cache(maxsize=_LRU_CACHE_SIZE_CALIB)
    def _sigma_vec(self, asset : str):
        """ Calibrated sigma vector, depends on the black_vol_calibration above.

        :param asset: asset to calculate sigma vector over.
        """

        nbf = self.nb_factors_for_asset(asset)  # number of factors

        return self.black_vol_calibration(asset).xf[nbf:(2*nbf)]

    @lru_cache(maxsize=_LRU_CACHE_SIZE_CALIB)
    def __factorCorrMatList(self, asset : str):
        """ Factor correlation matrix

        """

        nbf = self.nb_factors_for_asset(asset)

        return self._construct_corr_asset(asset, self.black_vol_calibration(asset).xf[(2*nbf):])

    # TODO: REWRITE THIS LRU CACHE FUNCTION
    @lru_cache(maxsize=_LRU_CACHE_SIZE_CALIB)
    def _betaT( self
              , asset : str
              , tenorList : List[datetime.date]):
        """
        Adjusts beta_T so that the atm vol is fitted perfectly.
        (assuming that kappa, sigma, rho has already been calibrated).
        The results are memoized.

        :param asset: name of the asset calibrated (e.g. 'WTI')
        :param tenorList: list of tenors for which the beta is calibrated
        """


        return self._com_vol_curves[asset].atmVol(tenorList) / \
               np.array([ self.black_vol( asset
                                        , self._kappa_vec(asset)
                                        , self._sigma_vec(asset)
                                        , self.__factorCorrMat(asset, asset)
                                        LLL
                                        , forward_idx)
                         for forward_idx in range(self.forward_curve_len[asset_nb])])

    # def check_black_vol_calib(self, asset_nb : str, reportingDiff=1e-2) -> None:
    #     """
    #     Checks the black vol calibration, logs the results if the calibration failed.
    #
    #     :param asset_nb: asset to be checked, e.g. 'WTI'
    #     :param reportingDiff: difference between model and market vols to be reported.
    #     """
    #
    #     model_atm_vols = np.array([self.black_vol(asset_nb
    #                                               , self._kappa_vec(asset_nb)
    #                                               , self._sigma_vec(asset_nb)
    #                                               , self.__factorCorrMat(asset_nb, asset_nb), fwd)
    #                                for fwd in range(self.forward_curve_len[asset_nb])])
    #
    #     diff = scipy.linalg.norm(model_atm_vols - self.atm_vol_list[asset_nb])
    #
    #     if diff > reportingDiff:
    #         logger.info('Calibration of ATM vols for asset nb. {0} FAILED. Diff= {1}'.format(asset_nb, str(diff)))
    #
    #     logger.debug('Calibration of ATM vols for asset nb. {0} succeeded. Diff = {1}'.format(asset_nb, str(diff)))

    # def __default_corr_mat(self, asset_nb : str, exp_nb : float) -> np.ndarray:
    #     """ Constructs the default correlation matrix
    #     the closer exp_nb is to 0, the more singular the matrix is
    #     and the correlation between forwards is closer to 1
    #     (does not need optimization)
    #
    #     :param asset_nb: asset in the model to be considered ('WTI')
    #     :param exp_nb: correlation number for the correlation matrix.
    #
    #     """
    #
    #     nb_tenors = len(self.fwd_curves[asset_nb])
    #
    #     return [np.exp(-(np.abs(j-i)*exp_nb))
    #             for i in range(nb_tenors)
    #                 for j in range(nb_tenors)]

    def black_corr_within_curve (self, asset_nb : str, ind_1 : int, ind_2 : int):
        """
        the cummulative correlation between the ind_1-th and the ind_2-th future's contract
        up to the option time of the smallest of the two contracts

        :pparam asset_nb: asset nb. for this curve

        """

        # TODO: opt_mat is WRONG
        opt_mat = self.optionTenorForFwdTenor(asset_nb, np.min(ind_1, ind_2))  # opt_mat until the smallest one
        corr = self.__factorCorrMat(asset_nb, asset_nb)  # correlation matrix
        kv = self._kappa_vec(asset_nb)
        sv = self._sigma_vec(asset_nb)
        ft = self.forward_tenors_list[asset_nb]  # forward vector
        nbf = self.nb_factors_for_asset(asset_nb)

        a = np.array([corr[ind_1, ind_2] * sv[factor_nb_1] * sv[factor_nb_2] *
                      np.product(self.__betaTList[asset_nb] ) *
                      (np.exp(- kv[factor_nb_1] * (ft[ind_1] - opt_mat) -
                              kv[factor_nb_2] * (ft[ind_2] - opt_mat) ) -
                       np.exp(- kv[factor_nb_1] * ft[ind_1] -
                              kv[factor_nb_2] * ft[ind_2])) /
                      (kv[factor_nb_1] + kv[factor_nb_2])
                      for factor_nb_1 in range(nbf)
                      for factor_nb_2 in range(nbf)])

        return (np.exp(np.sum(a)) - 1.) / self.black_vol(asset_nb, kv, sv, ind_1) / \
               self.black_vol(asset_nb, kv, sv, ind_2)  # covariance divided by 2 standard deviations

    def black_corr_within_curve_estimation(self, asset_nb, emp_corr_matrix):
        """
        Calibration of TODO within the curve.

        :param asset_nb: commodity asset
        :type asset_nb: int
        :param emp_corr_matrix: empirical correlation matrix
        :type emp_corr_matrix: 2-dim np.array
        """

        return scipy.optimize.brute(np.sum(np.sum(([self.black_corr_within_curve(asset_nb, ind_1, ind_2)
                                                     for ind_1 in range(self.forward_curve_len[asset_nb])
                                                         for ind_2 in range(self.forward_curve_len[asset_nb])]
                                                   - emp_corr_matrix)** 2))
                                   , 1)

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

        t_1 = self.optionTenorForFwdTenor(curve_1, tenor_1)
        t_2 = self.optionTenorForFwdTenor(curve_2, tenor_2)
        opt_mat = t_1 if  t_1 <= t_2 else t_2  # opt_mat until the smallest one
        kv1 = self._kappa_vec(curve_1)
        kv2 = self._kappa_vec(curve_2)

        bv1 = np.sqrt(self.V_fct_current(curve_1, tenor_1, opt_mat))  # square of integrated variance
        bv2 = np.sqrt(self.V_fct_current(curve_2, tenor_2, opt_mat))

        return sum([model_corr_mtx[factor_nb_1, factor_nb_2] *
                    self._sigma_vec(curve_1)[factor_nb_1] * self._sigma_vec(curve_2)[factor_nb_2] *
                    self.__betaTList[curve_1][tenor_1] * self.__betaTList[curve_2][tenor_2] *
                    np.exp(- kv1[factor_nb_1] * self.__difference_to_market_date(tenor_1)
                           - kv2[factor_nb_2] * self.__difference_to_market_date(tenor_2)) *
                    (np.exp((kv1[factor_nb_1] + kv2[factor_nb_2]) * opt_mat) - 1) /
                    (kv1[factor_nb_1] + kv2[factor_nb_2] )
                    for factor_nb_1 in range(self.nb_factors_for_asset(curve_1))
                    for factor_nb_2 in range(self.nb_factors_for_asset(curve_2))]) / (bv1 * bv2)

    def black_corr_intra_curves_factors( self
                                       , model_corr_mtx
                                       , curve_1     : str
                                       , curve_2     : str
                                       , tenor_1     : datetime.date
                                       , tenor_2     : datetime.date
                                       , factor_nb_1 : int
                                       , factor_nb_2 : int
                                       , opt_mat ):
        """
        same as function above (black_corr_intra_curves), but the factors are exposed
        :param curve_1: forward curve 1
        :param curve_2: forward curve 2
        tenor_1 and tenor_2 are tenor indices
        factor_nb_1, factor_nb_2 are factors for the two assets
        opt_mat ... until what maturity this is

        """

        kv1 = self._kappa_vec(curve_1)
        kv2 = self._kappa_vec(curve_2)
        sv1 = self._sigma_vec(curve_1)
        sv2 = self._sigma_vec(curve_2)
        bv1 = np.sqrt(self.__VOneFactor(curve_1, factor_nb_1, tenor_1, 0., opt_mat))
        bv2 = np.sqrt(self.__VOneFactor(curve_2, factor_nb_2, tenor_2, 0., opt_mat))

        return model_corr_mtx[factor_nb_1, factor_nb_2] * \
               sv1[factor_nb_1] * sv2[factor_nb_2] * \
               self.__betaTList[curve_1][tenor_1] * self.__betaTList[curve_2][tenor_2] * \
               np.exp(- kv1[factor_nb_1] * self.__difference_to_market_date(tenor_1) - kv2[factor_nb_2] * self.__difference_to_market_date(tenor_2)) * \
               (np.exp((kv1[factor_nb_1] + kv2[factor_nb_2]) * opt_mat) - 1.) / \
               (kv1[factor_nb_1] + kv2[factor_nb_2]) / (bv1 * bv2)

    def black_corr_intra_curves_calib( self
                                     , curve_1 : str
                                     , curve_2 : str
                                     , solver = 'scipy_cobyla'):
        """ Calibrates the intra-curve correlations.

        :param curve_1: curve 1 to for correlation calibration.
        :param curve_2: curve 2 for calibration.
        :param solver: solver to use in the OpenOpt
        """

        black_corr_intra_curve_vector = lambda model_corr_mtx, curve_1, curve_2, corr_len: \
            np.array([self.black_corr_intra_curves(model_corr_mtx,
                                                   curve_1, curve_2,
                                                   tenor, tenor)
                      for tenor in range(corr_len)])
            
        black_corr_intra_curve_vector_optim = lambda model_corr_mtx, curve_1, curve_2, corr_len: \
            scipy.linalg.norm(black_corr_intra_curve_vector(model_corr_mtx, curve_1, curve_2, corr_len) -
                              self.market_corr_list[curve_1][curve_2])

        # !!!!! CHECK CHECK CHECK 
        # RELATIONSHIP BETWEEN size_1 and size_2 
        corr_len_real = len(self.market_corr_list[curve_1][curve_2])
        # !!!! WRONG WRONG - 2,2 SHOULD BE REMOVED IN MULTIPLE PLACES BELOW <- THIS HAS BEEN CORRECTED, CHECK IF IT WORKS
        curve_1_nb_fact = self.nb_factors_for_asset[curve_1]  # this used to be 2
        curve_2_nb_fact = self.nb_factors_for_asset[curve_2]  # this used to be 2
        optim_pr = NSP(lambda corr_mtx_ravel: black_corr_intra_curve_vector_optim(corr_mtx_ravel.reshape ((curve_1_nb_fact,curve_2_nb_fact)),
                                                                                  curve_1, curve_2, corr_len_real),
                       self.__factorCorrMat(curve_1, curve_2).ravel(),
                       lb = self.__factorCorrMatLB(curve_1, curve_2).ravel(),
                       ub = self.__factorCorrMatUB(curve_1, curve_2).ravel())\
                      .solve(solver)
        # TODO: HERE IMPROVE!!!

        self.__factorCorrMatList[curve_1][curve_2] = self.__factorCorrMatList[curve_2][curve_1] = \
            np.array(optim_res.xf).reshape((curve_1_nb_fact, curve_2_nb_fact))  # assigning the matrix

        return np.array(optim_res.xf).reshape((2, 2))  # TODO: WHAT IS THIS 2 HERE??

    def __deltas_to_strikes(self
                            , asset : str
                            , tenorDate : datetime.date
                            , delta_vec_list : np.array) -> np.array:
        """
        Converts deltas to strikes for particular asset and tenor.

        :param asset: commodity asset to compute
        :param tenorDate: tenor considered.
        :returns: a vector of deltas from the strikes given in self.delta_vec_list
        """

        integrated_vol = self._com_vol_curves[asset].atmVol(tenorDate) * \
                         np.sqrt(self.__difference_to_market_date(self.optionTenorForFwdTenor(asset, tenorDate)))

        return np.exp((scipy.stats.norm.ppf(delta_vec_list) - 0.5 * integrated_vol ) * integrated_vol) * \
               self._com_fwd_curves[asset].getFwdValue(tenorDate)

    @staticmethod
    def __integr_analy( real_roots_tsf
                      , nb_real_roots  : int
                      , Asigma
                      , A0
                      , A1
                      , A2
                      , A3
                      , A4
                      , V):
        """ Integrate the polynomial between the roots, used for option calibration.

        :param real_roots_tsf:
        :param nb_real_roots: number of real roots

        """

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
                else:  # A3 < 0
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

        A0, A1, A2, A3, A4, V = self.__unpack_params(A_V, call_put_ind, strike)

        return scipy.integrate.quad(lambda x: np.max([A0 + A1 * V * x + A2 * V**2 * x**2 +
                                                      A3 * V**3 * x**3 + A4 * V**4 * x**4, 0.]) / \
                                              np.sqrt(2. * np.pi) * np.exp(- x**2 / 2.)
                                   , -np.inf
                                   , np.inf)[0]

    @staticmethod
    def __unpack_params(A_V, call_put_ind, strike):
        """ Unpacks the parameters A, V from A_V.

        """

        A, V = A_V[0:5], A_V[5]  # V= integrated volatility, not variance

        if call_put_ind == 1:  # call
            A0 = A[0] - strike
            A1, A2, A3, A4 = A[1:5]
        else:  # put
            A0 = strike - A[0]
            A1, A2, A3, A4 = - A[1:5]

        return A0, A1, A2, A3, A4, V

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
                poly_roots = np.sort(quartic_cy.QuadRoots(np.array([A2, A1, A0])))
            elif A4 == 0.:
                poly_roots = np.sort(quartic_cy.CubicRoots(np.array([A3, A2, A1, A0])))
            elif np.abs(A4) < 1e-6:
                poly_roots = np.sort(np.poly1d([A4, A3, A2, A1, A0]).roots)
            else:
                poly_roots = np.sort(quartic_cy.QuarticRoots(np.array([A4, A3, A2, A1, A0])))

        real_roots = poly_roots[poly_roots == poly_roots.real].real  # real roots only
        Asigma = np.array([A0, A1, A2, A3, A4]) * np.array([1., V, V**2, V**3, V**4])  # A multiplied by sigmas

        #if debug_mode:  # debug, selects the numeric approach
        #    return disc_fact * self.__integr_num(Asigma, call_put_ind, strike)
        #else:  # production mode

        return self.DF(ttm) * \
                   ComSkew.__integr_analy( real_roots / V
                                         , np.sum(poly_roots == poly_roots.real)  # number of real roots
                                         , Asigma
                                         , A0
                                         , A1
                                         , A2
                                         , A3
                                         , A4
                                         , V )

    def model_vol_surface(self, asset : str, C_vec, fwdDate : datetime.date):
        """ Computes model vols for asset_nb, C_vec, fwd_idx.

        :param asset: commodity considered.
        :param C_vec: skew vector
        """

        strikes = self.__deltas_to_strikes(asset, fwdDate)  # TODO: fix this part
        cp_ind = np.array([1 if (strike >= self._com_fwd_curves[asset].getFwdValue(fwdDate)) else -1 for strike in strikes])

        return np.array([vols.vols.black_vol_inverse( self._com_fwd_curves[asset].getFwdValue(fwdDate)
                                                    , strike
                                                    , opt_price
                                                    , self.optionTenorForFwdTenor(asset, fwdDate)
                                                    , self.DF(self.__difference_to_market_date(self.optionTenorForFwdTenor(asset, fwdDate)))
                                                    , cp
                                                    , self.blackVolInverseTol)
                         for opt_price, strike, cp in zip( [self.polynomial_european(asset, C_vec, fwdDate, strike, cp)
                                                            for strike, cp in zip(strikes, cp_ind)]
                                                         , strikes
                                                         , cp_ind ) ] )

    def __opt_fct_skew( self
                      , asset_nb
                      , fwdDateList : List[datetime.date] ):
        """
        Optimization function to minimize over the range 0: nb_tenors

        """

        # penalize the calibrated funtion for values of C where positive forward prices.
        # penalization level is 10000
        #    imp_vol_vec_model - self.vol_surface_list[asset_nb][fwd_idx, :]
        return NLP( lambda C_vec: scipy.linalg.norm(self.model_vol_surface(asset_nb, C_vec, fwdDateList) -
                                                    self._com_vol_curves[asset_nb].getVolForDate(fwdDateList))
                  , np.array([1., 0., 0.])  # TODO: THIS HAS TO BE IMPROVED
                  , iprint = -1 )\
                  .solve(ComSkew.NLP_SOLVER).xf

    @lru_cache(maxsize=100)  # TODO: FIX THIS HERE!!!
    def __c_vec_list(self
                     , asset_nb    : int
                     , fwdDateList : List[datetime.date]
                     , multiThreadInd = False):
        """
        Returns the skew parameters for asset_nb. If done the first time, it calibrates the parameters,
        otherwise it returns the stored value.

        :param asset_nb: the asset nb. to calibrate, such as 'wti'
        :param multiThreadInd: indicator whether to use multiple threads
        """

        if not multiThreadInd:
            return np.array(self.__opt_fct_skew(asset_nb, fwdDateList))

        # TODO: CHECK IF THIS CAN BE WRITTEN AS ENVIRONMENT
        # multithreading present
        pool = mp.Pool(processes=mp.cpu_count())
        curr_nb_tenors = len(fwdDateList)
        C = pool.map(opt_fct_skew_wrap,
                     zip([self] * curr_nb_tenors,
                         [asset_nb] * curr_nb_tenors,
                         range(curr_nb_tenors)))
        pool.close()
        return np.array(C)

    def __generate_large_corr_mat(self, nb_steps=300) -> None:
        """  Generates the factor correlation matrix from a list of list of corr. matrices
        gathered in __factorCorrMatList

        :param nb_steps: number of steps to converge to the correlation matrix.
        :returns: None, just sets the self.__completeCorrMat
        """

        totalNbFactors = sum([self.nb_factors_for_asset(asset) for asset in self.fwd_curves])
        self.__completeCorrMat = np.zeros((totalNbFactors, totalNbFactors))

        # sets the large correlation building blocks
        for asset_1 in self.fwd_curves:
            for asset_2 in self.fwd_curves:
                self.__completeCorrMat[self.__factor_positions(asset_1), self.__factor_positions(asset_2)] = self.__factorCorrMat(asset_1, asset_2)

        # find the closest matrix that is positive semidefinite
        self.__completeCorrMat = near_corr.near_corr_simple(self.__completeCorrMat, nb_steps) # !!!! 30 STEPS IS THIS ENOUGH
        while not (np.linalg.eig(self.__completeCorrMat)[0] > 0.).all():
            d1, v1 = np.linalg.eig(self.__completeCorrMat)
            d1p = np.diag(np.maximum(d1, 1.e-16))
            self.__completeCorrMat = np.dot(v1, np.dot(d1p, v1.transpose()))

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
            return self.__VOneFactor(asset_nb, i, fwd_idx, t_prev, t_next)

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
        0-th dimension: asset_nb
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

        for comCurve in self.fwd_curves:
            sim_curves_shape = (len(simulation_times), len(tenor_list), nb_simulations)
            simulated_curves[comCurve] = np.empty(sim_curves_shape)  #  if not cuda_ind else gpa.zeros(sim_curves_shape, dtype=rn_type)
            fwd_c_col[comCurve] = self._com_fwd_curves[comCurve].getFwdValues(tenor_list)
            simulated_curves[comCurve][0, :, :] = fwd_c_col


        X      = [np.zeros((len(fwd_c_col[comCurve]), nb_simulations)) for comCurve in self.fwd_curves]
        X_prev = [np.empty ((len(fwd_c_col[comCurve]), nb_simulations)) for comCurve in self.fwd_curves]

        nb_factors = np.sum(self.nb_factors_for_asset)

        # looping over time steps
        #   simulates ln process, basis for skew as well
        #   t_i ... idx of sim_time
        #   fact_sum ... factors of the individual assets
        nb_time_steps = len(simulation_times)
        for t_i in range(nb_time_steps):
            simulated_rn = ComSkew.__simulate_std_normal( nb_factors
                                                        , self.__completeCorrMat
                                                        , nb_simulations )

            for fwd_curve in self.fwd_curves:
                asset = fwd_curve.name
                nb_factors_asset = self.nb_factors_for_asset(asset)
                old_cov_mat = self.__completeCorrMat[self.__factor_positions(asset), self.__factor_positions(asset)]
                tenor_used = tenor_list[asset]
                old_chol_inv = np.linalg.inv(np.linalg.cholesky(old_cov_mat))
                sims_Z_unit = np.dot(old_chol_inv, simulated_rn[:, self.__factor_positions(asset)].transpose())

                for tenor_idx, tenor_nb in enumerate(tenor_used):
                    # prepare cov mtx
                    cov_chol = np.linalg.cholesky(np.array([[self._var_covar_mtx(asset, tenor_nb, i, j, t_i, self.simulation_times)
                                                             for j in range(nb_factors_asset)]
                                                            for i in range(nb_factors_asset)]))

                    delta_X = np.sum(np.dot(cov_chol, sims_Z_unit), axis=0)
                    # quadratic variation of delta_X, also q_v = V_u
                    qv = np.sum([[self._V_cross_factor( asset
                                                      , factor_1
                                                      , factor_2
                                                      , tenor_nb
                                                      , tenor_nb
                                                      , 0. if t_i == 0 else self.simulation_times[t_i - 1]
                                                      , self.simulation_times[t_i])
                                  for factor_1 in range(nb_factors_asset)]
                                 for factor_2 in range(nb_factors_asset)])

                    X_prev[asset][tenor_idx, :] = X[asset][tenor_idx, :]
                    X[asset][tenor_idx, :] = X_prev[asset][tenor_idx, :] + delta_X

                    # F_res = F_u * (1. + X_u + 0.5 * c1 * (X_u**2 - V_u) +
                    #                c2 * (X_u**3 - 3. * X_u * V_u) / 6. +
                    #                c3 * (X_u**4 - 6. * V_u * X_u**2 + 3. * V_u**2) / 24.)
                    # self.simulated_curves[asset_nb][t_i, tenor_idx, :] = F_res
                    c1, c2, c3 = self._CVecList[asset][tenor_nb, :]
                    opd_avx.skew_fom( self._com_fwd_curves[asset][tenor_nb]
                                    , X[asset][tenor_idx, :]  # delta_X
                                    , 0.5 * c1
                                    , qv  # V_u, quadratic variation
                                    , c2/6.
                                    , c3/24.
                                    , self.simulated_curves[asset][t_i, tenor_idx, :]
                                    , nb_simulations )

        return self.simulated_curves  # TODO: To correct, simulated_curves should be private

    def simulate_1nb( self, nb_simulations : int
                    , simulation_times = None
                    , set_seed         = None ) -> Dict[str]:
        """ Simulate the 1NB (rolling) contract.

        # TODO: FIX THIS HERE!!
        generates a 3-dimensional array:
          0-th dimension: asset_nb
          1-st dimension: simulation times
          2-rd dimension: repeats of the curve

        :param nb_simulations: number of simulations to simulate.
        :param simulation_times: times when to simulate curves, if None TODO: WHAT THEN???
        :param set_seed: set the seed for simulations.
        """

        assert simulation_times[-1] > self.fwd_curves[0][-1], \
            'Last simulation time is larger than the largest forward tenor.'

        sim_times = simulation_times if simulation_times else self.simulation_times()

        simulated_curves = self.simulate_curves(nb_simulations, sim_times, set_seed)
        new_simulated_curves = {}

        for fwd_curve in self.fwd_curves:
            asset = fwd_curve.name
            new_simulated_curves[asset] = np.empty((len(sim_times), nb_simulations))
            for t_i in range(self.nb_time_steps):
                current_nb = np.sum(self.fwd_curves[asset] <= self.simulation_times[t_i])
                new_simulated_curves[asset][t_i, :] = simulated_curves[asset][t_i, current_nb,:]

        return new_simulated_curves

    @lru_cache(maxsize=_LRU_CACHE_SIZE_CALIB)  # TODO: THIS IS NOT RIGHT HERE!!!
    def __factor_positions(self, asset_nb : int) -> slice:
        """
        Returns factor positions in a matrix for asset_nb
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
                            , tenors_list
                            , seed = None ):
        """ Simulate first of month (fom) curves.

        generates a list of 2 dim arrays:
           1-st dim: tenor
           2-nd dim: simulation

        :param asset: asset for which to simulate, e.g. 'WTI'
        :param nb_simulations: nb. of simulations
        :param sim_times: simulation times
        :param tenors_list: list of tenors to simulate TODO: LOTS TO DISCUSS HERE!!!
        """

        np.random.seed(seed=seed)

        fwd_c_len = len(tenors_list)
        sim_times = self.option_tenors_list[asset][tenors_list]

        sim_fom = np.empty((fwd_c_len, nb_simulations))

        # looping over tenors
        #    t_i ... idx of sim_time (also tenor)
        #    fact_sum ... factors of the individual assets
        for tenor_idx, tenor_date in enumerate(tenors_list):
            tenor_nb = tenor_date
            t_curr = sim_times[tenor_idx]
            F_curr = self.fwd_curves[asset][tenor_nb]
            nb_factors = np.sum(self.nb_factors_for_asset)  # total nb. of factors
            simulated_rn = np.random.multivariate_normal(np.zeros(nb_factors), self.__completeCorrMat,
                                                         size=nb_simulations)
            nb_factors_asset = self.nb_factors_for_asset[asset]
            new_cov_mat = np.array([[self._var_covar_mtx(self, asset, tenor_nb, i, j, tenor_idx, sim_times)
                                    for j in range(nb_factors_asset)]
                                    for i in range(nb_factors_asset)])

            new_chol = np.linalg.cholesky(new_cov_mat)
            old_cov_mat = self.__completeCorrMat[self.__factor_positions(asset), self.__factor_positions(asset)]
            old_chol = np.linalg.cholesky(old_cov_mat)

            sims_Z = simulated_rn[:, self.__factor_positions(asset)].transpose()
            sims_Z_unit = np.dot(np.linalg.inv(old_chol), sims_Z)
            delta_X = np.sum(np.dot(new_chol, sims_Z_unit), axis=0)
            qv = np.sum([[self._V_cross_factor(asset, factor_1, factor_2, tenor_nb, tenor_nb, 0., t_curr)
                          for factor_1 in range(nb_factors_asset)]
                         for factor_2 in range(nb_factors_asset)])

            if self.model_skew_ln_ind is 'ln_ln':
                sim_fom[tenor_idx, :] = F_curr * np.exp(delta_X - 0.5 * qv)
            else:
                c0, c1, c2 = self._CVecList[asset][tenor_nb, :]
                # sim_fom[t_i, :] = F_curr * \
                #                    (1. + delta_X + 0.5 * c1 * (delta_X**2 - qv) +
                #                    c2 * (delta_X**3 - 3. * delta_X * qv) / 6. +
                #                    c3 * (delta_X**4 - 6. * qv * delta_X**2 + 3. * qv**2) / 24.)
                opd_avx.skew_fom( F_curr
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
            np.sqrt(self.__difference_to_market_date(self.optionTenorForFwdTenor(asset, fwd_date)))
        f0t = self.fwd_curves[asset].fwdValue(fwd_date)

        return ( (1. - cc1 * v**2 / 2. + cc3 * v**4 / 8.) * f0t
               , (1. - cc2 * v**2 / 2.) * f0t
               , (cc1 / 2. - cc3 * v**2 / 4.) * f0t
               , (cc2 / 6.) * f0t
               , (cc3 / 24.) * f0t
               , v )
