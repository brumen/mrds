#   skew model used for tolling.
#

import datetime
import numpy as np
import logging

from typing import List, Tuple

from config import work_dir, CUDA_PRESENT

from mrds          import ComSkew
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

# if CUDA_PRESENT:
#     F_skew_el = open(work_dir + 'cuda/skew_tsf.c', 'r').read()
#     F_skew_mod = SourceModule(F_skew_el)
#     F_skew_fct = F_skew_mod.get_function('F_skew_tsf')


class ComSkewTolling(ComSkew):
    """ Adds the methods responsible only for tolling simulation, etc.
    """

    def __init__(self
                 , mkt_date        : datetime.date
                 , fwd_curves      : List[FwdCurve]
                 , vol_curves      : List[Volatility]
                 , days_partition  : List[List[int]]
                 , hours_partition : List[List[int]]
                 , discount_curve = None
                 , calc_date      = None ):

        """ Initialization of the skew model for tolling simulation.

        :param mkt_date: market date
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, in case they are different than forward curves.
        :param discount_curve: discount curve, a function of fwd_date, returns lambda fwd_date: discount(mkt_date, fwd_date)
        :param calc_date: calculation date.
        :param days_partition: partition of days,  Mon = 0, Sun = 6, e.g. [[0,1,2,3,4], [5,6]]  # TODO: MAYBE CHANGE THIS
        :param hours_partition: partition of hours for each block, e.g [[8,16], [16, 8]]
        """

        super().__init__(mkt_date, fwd_curves, vol_curves, discount_curve=discount_curve, calc_date=calc_date)

        self.days_partition  = days_partition
        self.hours_partition = hours_partition

    def generate_days_vecs(self, cuda_ind=False) -> Tuple:
        """ Generate days for simulate_spot_blocks.

        :param cuda_ind: Whether to use and generate objects on cuda, or on cpu.
        :returns: tuple of days TODO: FIX THIS
        """

        # construct the equiv. of days = range(31)/365.25
        days = np.array([0.])
        for day in range(31):  # all possible days
            day_week = np.mod(day, 7)
            hours_for_day_week = [hp for (hp, dp) in zip(self.hours_partition, self.days_partition)
                                  if day_week in dp][0]
            days = np.append(days, days[-1] + np.cumsum(hours_for_day_week)/24./365.25)
        days_d = gpa.to_gpu(days)

        if cuda_ind:
            days_diff = gpa.empty(len(days))
            days_diff[0] = np.array(0.)
            days_diff[1:] = np.diff(days)  # TODO: PROBABLY THIS IS INEFFICIENT, CHECK!!
        else:
            days_diff = np.empty(len(days))
            days_diff[0] = 0.
            days_diff[1:] = np.diff(days)

        return days, days_d, days_diff

    def simulate_spot_blocks( self
                            , assets          : List[str]
                            , nb_simulations  : int
                            , tenors_chosen = None
                            , set_seed      = None
                            , cuda_ind      = False ):
        """ Same as simulate_spot_blocks, but for all blocks. TODO: DESCRIBE THIS BETTER

        :param assets: list of assets to which asset to
        :returns:
        """

        days_tuple = self.generate_days_vecs(cuda_ind=cuda_ind)

        # construct the equiv. of days = range(31)/365.25
        days, days_d, days_diff, days_diff_l = days_tuple
        fom_sims_all = self.simulate_1nb(assets, nb_simulations, tenors_chosen, set_seed=set_seed)

        for asset in assets:
            self.gen_days_number(asset)  # TODO: FIX THIS HERE!!!
            self.gen_spot_rn(nb_simulations, cuda_ind=cuda_ind)

        spot_sims = {}
        if not tenors_chosen:
            cv_tenors = zip(range(len(self.cash_vol_list(asset))), self.cash_vol_list(asset))
        else:
            cv_tenors = zip(tenors_chosen, self.cash_vol_list(asset)[tenors_chosen])

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
        days, days_d, days_diff, days_diff_l = self.generate_days_vecs(cuda_ind=cuda_ind)
        self.gen_spot_rn(nb_simulations, cuda_ind=cuda_ind)

        fom_sims_all = self.simulate_spot_blocks( assets
                                                , nb_simulations
                                                , tenors_chosen = tenors_chosen
                                                , cuda_ind      = cuda_ind )

        for asset in assets:
            self.gen_days_number(asset)
            cv_m = self.cash_vol_list[asset][month]
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
