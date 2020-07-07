#   skew model used for tolling.
#

import datetime
import numpy as np
import logging

from typing import List, Tuple, Dict

from mrds.config   import CUDA_PRESENT
from mrds_spot     import ComSkewSpot
from vols.vols     import Volatility
from forward_curve import FwdCurve

if CUDA_PRESENT:
    import pycuda.curandom
    import pycuda.gpuarray as gpa
    import pycuda.cumath
    # from pycuda.compiler import SourceModule
    from cuda import cuda_ops
    from cuda.cuda_ops import matmul


logger = logging.Logger(__name__)



class ComSkewTolling(ComSkewSpot):
    """ Adds the methods responsible only for tolling simulation, etc.
    """

    def __init__(self
                 , mkt_date        : datetime.date
                 , fwd_curves      : List[FwdCurve]
                 , vol_curves      : List[Volatility]
                 , cash_vol_curves : List[Volatility]
                 , days_partition  : List[List[int]]
                 , hours_partition : List[List[int]]
                 , discount_curve = None
                 , calc_date      = None ):

        """ Initialization of the skew model for tolling simulation.

        :param mkt_date: market date
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, in case they are different than forward curves.
        :param cash_vol_curves: cash vol curves, corresponding to fwd_curves & vol_curves
        :param discount_curve: discount curve, a function of fwd_date, returns lambda fwd_date: discount(mkt_date, fwd_date)
        :param calc_date: calculation date.
        :param days_partition: partition of days,  Mon = 0, Sun = 6, e.g. [[0,1,2,3,4], [5,6]]  # TODO: MAYBE CHANGE THIS
        :param hours_partition: partition of hours for each block, e.g [[8,16], [16, 8]]
        """

        super().__init__(mkt_date, fwd_curves, vol_curves, cash_vol_curves, discount_curve=discount_curve, calc_date=calc_date)

        self.days_partition  = days_partition
        self.hours_partition = hours_partition

    def __F_skew_tsf_cuda(self):
        # TODO: FIX work_dir here
        with open(work_dir + 'cuda/skew_tsf.c', 'r') as F_skew_el:
            return SourceModule(F_skew_el.read()).get_function('F_skew_tsf')

    def _generate_days_vecs(self, cuda_ind=False) -> Tuple:
        """ Generate days for simulate_spot_blocks.

        :param cuda_ind: Whether to use and generate objects on cuda, or on cpu.
        :returns: tuple of days TODO: FIX THIS
        """

        # construct the equiv. of days = range(31)/365.25
        days = np.array([0.])
        for day in range(31):  # all possible days  # TODO: THIS IS WRONG 31 HERE
            day_week = np.mod(day, 7)
            hours_for_day_week = [hp for (hp, dp) in zip(self.hours_partition, self.days_partition)
                                  if day_week in dp][0]
            days = np.append(days, days[-1] + np.cumsum(hours_for_day_week)/24./365.25)

        # gpu days
        if cuda_ind:
            days_diff = gpa.empty(len(days))
            days_diff[0] = np.array(0.)
            days_diff[1:] = np.diff(days)  # TODO: PROBABLY THIS IS INEFFICIENT, CHECK!!

            return gpa.to_gpu(days), days_diff

        # cpu generation
        days_diff = np.empty(len(days))
        days_diff[0] = 0.
        days_diff[1:] = np.diff(days)

        return days, days_diff

    def simulate_spot_blocks( self
                            , assets         : List[str]
                            , nb_simulations : int
                            , tolling_start  : datetime.date
                            , tolling_end    : datetime.date
                            , set_seed      = None
                            , cuda_ind      = False ) -> Dict[str, np.ndarray]:
        """ Same as simulate_spot_blocks, but for all blocks. TODO: DESCRIBE THIS BETTER

        :param assets: list of assets to which asset to simulate block prices for.
        :param nb_simulations: number of simulations.
        :param tolling_start: start of the tolling simulations
        :param tolling_end: end of tolling sims.
        :param set_seed: optional param for debugging, so that simulations are always the same
        :param cuda_ind: indicator whether to do the computations on cuda.
        :returns: dictionary, where keys are simulated assets, and values are TODO: FINISH HERE!!!
        """

        # obtain the months corresponding to tolling_start and tolling_end.
        first_month = datetime.date(tolling_start.year, tolling_start.month, 1)  # first of first month
        last_month  = datetime.date(tolling_end.year  , tolling_end.month  , 1)  # first of last month

        months_to_use = 1  # TODO: dates of first of months between first_month and last_month



        days, days_diff = self._generate_days_vecs(cuda_ind=cuda_ind)

        # construct the equiv. of days = range(31)/365.25
        fom_sims_fom = self.simulate_1nb(assets, nb_simulations, months_to_use, set_seed=set_seed)

        spot_sims = {}
        for asset in assets:

            self.gen_spot_rn(nb_simulations, cuda_ind=cuda_ind)

            cash_curves_asset = self._cash_vol_curves(asset)
            # cash vol tenors
            cv_tenors = [cash_curves_asset.implied_vol(fwd_date, K, ttm)
                         for fwd_date in cash_curves_asset.vol_dates if not tenors_chosen else tenors_chosen]


            if cuda_ind:  # cuda usage
                w_days = pycuda.cumath.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l]
                cuda_ops.cumsum_cuda(w_days)
                for fwd_tenor_nb, cash_vol_tenor in cv_tenors:
                    # fom in column format
                    fom_sims = fom_sims_all[fwd_tenor_nb, :]   # row vec
                    mult_1 = np.float32(-0.5 * cash_vol_tenor**2)
                    mult_2 = np.float32(cash_vol_tenor)
                    col_vec = pycuda.cumath.exp(days * mult_1 + w_days * mult_2)
                    # transpose is used
                    spot_sims[asset][fwd_tenor_nb] = cuda_ops.vtpv(fom_sims, col_vec, tm_ind='t', transpose_ind=True).transpose()
            else:  # no cuda
                w_days = np.cumsum( np.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l]
                                  , axis=1)
                for fwd_tenor_nb, cash_vol_tenor in enumerate(cv_tenors):
                    # fom in column format
                    fom_sims_for_tenor = fom_sims_all[asset][fwd_tenor_nb, :]
                    fom_sims = fom_sims_for_tenor.reshape((len(fom_sims_for_tenor), 1))  # column vector
                    spot_sims[asset][fwd_tenor_nb] = np.transpose(fom_sims *
                                                           np.exp(-0.5 * cash_vol_tenor**2 * days +
                                                                  cash_vol_tenor * w_days))

        return spot_sims

    # TODO: CHECK WHY THIS IS NECESSARY
    def simulate_spot_blocks_2( self
                              , assets
                              , month
                              , nb_simulations
                              , tenors_chosen = None
                              , cuda_ind      = False ):
        """ Generates spot blocks of month m from fom_sims_all (used for a tolling model)

        :param m: month to simulate spot block from
        :type m: int
        """

        # construct the equiv. of days = range(31)/365.25
        days, days_d, days_diff, days_diff_l = self._generate_days_vecs(cuda_ind=cuda_ind)
        self.gen_spot_rn(nb_simulations, cuda_ind=cuda_ind)

        fom_sims_all = self.simulate_spot_blocks( assets
                                                , nb_simulations
                                                , tenors_chosen = tenors_chosen
                                                , cuda_ind      = cuda_ind )

        for asset in assets:
            self._number_days_for_month(asset)
            cv_m = self.cash_vol_curves(asset)[month]
            fom_sims_used = fom_sims_all[asset]

            if cuda_ind:  # cuda usage
                w_days = pycuda.cumath.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l]
                cuda_ops.cumsum_cuda(w_days)
                # fom in column format
                fom_sims = fom_sims_used[tenors_chosen.index(m), :]   # row vec
                mult_1 = np.float32(-0.5 * cv_m**2)
                mult_2 = np.float32(cv_m)
                col_vec = pycuda.cumath.exp(days_d * mult_1 + w_days * mult_2)
                # transpose is used
                spot_sims = cuda_ops.vtpv(fom_sims, col_vec, tm_ind='t', transpose_ind=True).transpose()

            else:  # no cuda
                w_days = np.cumsum(np.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l],
                                   axis=1)
                # fom in column format
                fom_sims = fom_sims_used[m, :].reshape((len(fom_sims_used[tenors_chosen.index(m), :]), 1))
                spot_sims = np.transpose(fom_sims * np.exp(-0.5 * cv_m**2 * days + cv_m * w_days))

        return spot_sims
