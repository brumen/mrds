#
#   skew model for Spot processes
#

from config import CUDA_PRESENT

import datetime as dt
import numpy as np
import calendar

from typing import List, Dict

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

from mrds import ComSkew

import logging


logger = logging.Logger(__name__)


class ComSkewSpot(ComSkew):
    """
    Building on the existing model

    """

    @property
    def cash_corr(self):
        return self._cash_corr

    @cash_corr.setter
    def cash_corr(self, cc):
        self._cash_corr = cc

    def set_cash_vols(self, asset_nb, cash_vols):
        """
        Sets the cash vols for the particular asset.

        """

        # TODO: FIX THIS HERE!
        assert len(cash_vols) == len(fwd_curve_list[asset_nb])

        self.cash_vol_list[asset_nb] = cash_vols

    def gen_days_number(self, asset_nb : int) -> Dict[int]:
        """
        Generates a dict of number of days per month for every tenor, saves it to
           self.nb_days_month and returns the same thing.

        :param asset_nb: which asset
        :returns: a dictionary where the key is the number of days for that month in the model,
                  i.e. if m = 0 that refers to the number of days for the first month generated
        :rtype: dict[int] = int
        """

        if not self.days_nb_const_ind:  # if not yet constructed

            nb_fwds = len(self.forward_tenors_list[asset_nb])
            self.nb_days_month = {}
            beg_curr_month = self.mkt_date - dt.timedelta(self.mkt_date.day - 1)
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

    def gen_spot_rn( self
                   , nb_simulations : int
                   , nb_days  = 65
                   , cuda_ind = False
                   , rn_type  = np.float32 ):
        """
        Returns the random walk of nb_simulations for nb_days.
        Nb_days can also be the number of blocks.

        :param nb_simulations: number of simulations.
        :param nb_days: number of days (or blocks) to simulate.
        :param cuda_ind: indicator for cuda.
        :returns:
        :rtype: np.array or gpa.GPUarray depending on what cuda_ind is.
        """

        if not self.simulate_spot_rn_ind:  # random walk not yet initialized
            self.spot_rn = {}

            for day_idx in range(nb_days):
                self.spot_rn[day_idx] = self.gen_cash_rns(nb_simulations, cuda_ind=cuda_ind)

            self.spot_rn_a = {}
            for asset_nb in range(self.nb_assets):
                self.spot_rn_a[asset_nb] = np.empty((nb_simulations, nb_days)) if cuda_ind==False else gpa.empty((nb_simulations, nb_days), dtype=rn_type)
                for day_idx in range(nb_days):
                    self.spot_rn_a[asset_nb][:, day_idx] = self.spot_rn[day_idx][:, asset_nb]

            self.simulate_spot_rn_ind = True

        else:
            if self.spot_rn[0].shape[0] != nb_simulations:
                for day_idx in range(nb_days):
                    self.spot_rn[day_idx] = self.gen_cash_rns(nb_simulations, rn_type=rn_type)
                    self.spot_rn_a = {}
                    for asset_nb in range(self.nb_assets):
                        self.spot_rn_a[asset_nb] = np.empty((nb_simulations, nb_days)) if cuda_ind==False else gpa.empty((nb_simulations, nb_days), dtype=rn_type)
                        for day_idx in range(nb_days):
                            self.spot_rn_a[asset_nb][:, day_idx] = self.spot_rn[day_idx][:, asset_nb]

    def gen_cash_rns( self
                    , nb_simulations
                    , rn_type=np.float32
                    , cuda_ind = False):
        """
        Generates the cash correlations

        """

        if cuda_ind:  # cuda version
            rng = pycuda.curandom.XORWOWRandomNumberGenerator()
            spot_rn_init = gpa.empty((self.nb_assets, nb_simulations), dtype=rn_type)
            cash_corr_gpu = gpa.to_gpu(np.linalg.cholesky(self.cash_corr).astype(rn_type))
            curand.gen_eff_dev_rns(spot_rn_init.size, np.longlong(spot_rn_init.ptr), rng)

            return cuda_ops.matmul(cash_corr_gpu, spot_rn_init)

        # cpu version
        return np.random.multivariate_normal(np.zeros(self.nb_assets)
                                            , self.cash_corr
                                            , size=nb_simulations)

    def simulate_spot(self, asset_nb, nb_simulations):
        """
        Simulate daily spot using for all tenors the cash_vols for asset asset

        """

        self.gen_days_number(asset_nb)
        self.gen_spot_rn(nb_simulations)
        self.simulate_curves(nb_simulations, self.option_tenors_list[asset_nb])
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
