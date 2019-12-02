#
#   skew model for Spot processes
#


import datetime
import numpy as np
import calendar
import logging

from config import CUDA_PRESENT
from typing import List, Dict, Union

# cuda (this can be imported even if cuda is not present)
if CUDA_PRESENT:
    import pycuda.curandom
    import pycuda.gpuarray as gpa
    import pycuda.cumath
    import curand  # TODO: THIS IS WRONG

import matplotlib as mpl
mpl.use('TkAgg')

if CUDA_PRESENT:
    from cuda import cuda_ops
    from cuda.cuda_ops import matmul

from mrds          import ComSkew
from vols.vols     import Volatility
from forward_curve import FwdCurve
from correlations  import corr_hyp_sec_two_fronts_time_diff


logger = logging.Logger(__name__)


class ComSkewSpot(ComSkew):
    """
    Building on the existing model

    """

    def __init__(self
                 , mkt_date        : datetime.date
                 , fwd_curves      : List[FwdCurve]
                 , vol_curves      : List[Volatility]
                 , cash_vol_curves : List[Volatility]
                 , cash_correlations = None
                 , discount_curve    = None
                 , calc_date         = None ):

        """ Initialization of the skew model for tolling simulation.

        :param mkt_date: market date
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, in case they are different than forward curves.
        :param cash_vol_curves: cash vol curves, corresponding to fwd_curves & vol_curves
        :param cash_correlations: cash correlation function between asset_1, asset_2, see _cash_correlation method below
        :param discount_curve: discount curve, a function of fwd_date, returns lambda fwd_date: discount(mkt_date, fwd_date)
        :param calc_date: calculation date.
        """

        super().__init__(mkt_date, fwd_curves, vol_curves, discount_curve=discount_curve, calc_date=calc_date)

        # new things in this class.
        self.__cash_vol_curves   = cash_vol_curves
        self.__cash_correlations = cash_correlations

    def _cash_vol_curves(self, asset : str) -> Volatility:
        """ Returns the cash vol curve for the particular asset. If you enter the
            wrong asset, None is returned.

        :param asset: asset cash vol to be returned.
        :returns: the Volatility subclass for that asset
        """

        for fwd_curve, cash_vol_curve in zip(self.fwd_curves, self.__cash_vol_curves):
            if fwd_curve.fwd_name == asset:
                return cash_vol_curve

    def _cash_correlation(self
                         , asset_1 : str
                         , asset_2 : str
                         , fwd_date_1 : datetime.date
                         , fwd_date_2 : datetime.date
                         , default_corr = 0.95 ):
        """ Cash correlations between asset_1 and asset_2

        :param asset_1: first asset to get correlations. ('ERCOT-PEAK')
        :param asset_2: second asset to compute correlations. ('ERCOT-OFFPEAK')
        :param fwd_date_1: date on the first curve (asset_1)
        :param fwd_date_2: date on the second curve
        :param default_corr: default correlation between asset_1 & asset_2, in hyp_sec form
        """

        if self.__cash_correlations:  # cash correlation is a given function
            return self.__cash_correlations(asset_1, asset_2, fwd_date_1, fwd_date_2)

        # else: default correlation
        return corr_hyp_sec_two_fronts_time_diff(default_corr, fwd_date_1, fwd_date_2)

    @staticmethod
    def _number_days_for_month(month_start_date : datetime.date) -> int:
        """ Generates a dict of number of days per month for every tenor.

        :param month_start_date: date of the start of that month
        :returns: number of days for that month
        """

        if month_start_date.month == 12:
            next_month_start = datetime.date(month_start_date.year + 1, 1, 1)
        else:
            next_month_start = datetime.date(month_start_date.year, month_start_date.month+1, 1)

        return (next_month_start - month_start_date).days

    @staticmethod
    def _cash_rns( cash_corr      : np.ndarray
                 , nb_simulations : int
                 , rn_type        = float
                 , cuda_ind = False ):
        """ Generates the cash correlations

        :param cash_corr: matrix of cash correlations
        :param nb_simulations: number of simulations.
        :param rn_type: type of random numbers to generate.
        :param
        """

        nb_assets = cash_corr.shape[0]  # cash_corr is a square matrix

        if cuda_ind:  # cuda version
            rng = pycuda.curandom.XORWOWRandomNumberGenerator()
            spot_rn_init = gpa.empty((nb_assets, nb_simulations), dtype=rn_type)
            cash_corr_gpu = gpa.to_gpu(np.linalg.cholesky(cash_corr).astype(rn_type))
            curand.gen_eff_dev_rns(spot_rn_init.size, np.longlong(spot_rn_init.ptr), rng)

            return cuda_ops.matmul(cash_corr_gpu, spot_rn_init)

        # cpu version
        return np.random.multivariate_normal( np.zeros(nb_assets)
                                            , cash_corr
                                            , size = nb_simulations )

    @staticmethod
    def __create_first_of_months(start_date : datetime.date, end_date : datetime.date):

        first_first_of_month = datetime.date(start_date.year, start_date.month, 1)

        if start_date.month == 12:
            next_first_of_month = datetime.date(start_date.year, 1, 1)
        else:
            next_first_of_month = datetime.date(start_date.year, start_date.month+1, 1)

        last_first_of_month = datetime.date(end_date.year, end_date.month, 1)

    def simulate_spot( self
                     , assets         : List[str]
                     , start_date     : datetime.date
                     , end_date       : datetime.date
                     , nb_simulations : int ):
        """ Simulate daily spots from start_date to end_date.

        :param assets: list of assets to simulate (['ERCOT_NORTH', 'ERCOT_SOUTH'])
        :param start_date: start of simulations
        :param end_date: end of simulations
        :param nb_simulations: number of simulations
        :returns: TODO;
        """

        # create first of months - then use simulate_1nb
        self.simulate_1nb(assets, nb_simulations, self.__create_first_of_months(start_date, end_date))

        #self._number_days_for_month(asset_nb)
        #self.gen_spot_rn(nb_simulations)
        spot_sims = {}
        for fwd_tenor_nb, cash_vol_tenor in enumerate(self.cash_vol_list[asset_nb]):
            nb_days_m = self.nb_days_month[fwd_tenor_nb]
            forward_tenors_dates = self.forward_tenors_list[asset_nb]
            if fwd_tenor_nb != (self.forward_curve_len[asset_nb] - 1):
                # not at the end of the month
                month_start, month_end = forward_tenors_dates[fwd_tenor_nb], forward_tenors_dates[fwd_tenor_nb+1]
            else:
                month_start = forward_tenors_dates[fwd_tenor_nb]
                month_end = month_start + 1./12.

            days = np.linspace(month_start, month_end, nb_days_m)
            day_len = days[1] - days[0]
            fom_sims = self.simulated_curves[asset_nb][fwd_tenor_nb, fwd_tenor_nb, :]
            W_days = np.cumsum(np.sqrt(day_len) * self.spot_rn[:, :nb_days_m], axis=1)
            spot_sims[fwd_tenor_nb] = np.transpose(fom_sims.reshape((len(fom_sims), 1)) *
                                                   np.exp(-0.5 * cash_vol_tenor**2 * days + cash_vol_tenor * W_days))

        return spot_sims
