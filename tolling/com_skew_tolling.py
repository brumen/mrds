#   skew model used for tolling.
#

import numpy as np
import logging

from typing import List

from config import work_dir, CUDA_PRESENT
from mrds   import ComSkew

if CUDA_PRESENT:
    import pycuda.curandom
    import pycuda.gpuarray as gpa
    import pycuda.cumath
    from pycuda.compiler import SourceModule
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

    @staticmethod
    def generate_days_vecs(hours_partition, days_partition, cuda_ind=False):
        """ Generate days for simulate_spot_blocks.

        :param hours_partition: partition of day in blocks
        :param days_partition: partition of the week in blocks, e.g. [[0,1,2,3,4], [5,6]] means that
                               days 0-4 are one block, and 5,6 are another block
        """

        # construct the equiv. of days = range(31)/365.25
        days = np.array([0.])
        for day in range(31):  # all possible days
            day_week = np.mod(day, 7)
            hours_for_day_week = [hp for (hp, dp) in zip(hours_partition, days_partition)
                                  if day_week in dp][0]
            days = np.append(days, days[-1] + np.cumsum(hours_for_day_week)/24./365.25)
        days_d = gpa.to_gpu(days)

        if cuda_ind:
            days_diff = gpa.empty(len(days))
            days_diff[0] = np.array(0.)
            days_diff[1:] = np.diff(days)  # TODO: WHY IS np. here ???
        else:
            days_diff = np.empty(len(days))
            days_diff[0] = 0.
            days_diff[1:] = np.diff(days)

        return days, days_d, days_diff

    def simulate_spot_blocks( self
                            , assets : List[str]
                            , nb_simulations
                            , days_partition
                            , hours_partition
                            , tenors_chosen = None
                            , set_seed      = None
                            , cuda_ind      = False ):
        """ Same as simulate_spot_blocks, but for all blocks.
        Simulates the spots from this model, used for a tolling model.

        :param asset: asset which to simulate.
        :param days_partition: a partition of days in the week, i.e. [[0, 1, 2, 3, 4], [5, 6]]
        :type days_partition: list[list[int]]
        :param hours_partition: [hours for blocks, e.g. [[6, 18], [12, 12]]
        :type hours_partition: list[list[int]]
        """

        days_tuple = self.generate_days_vecs(hours_partition, days_partition, cuda_ind=cuda_ind)

        # construct the equiv. of days = range(31)/365.25
        days, days_d, days_diff, days_diff_l = days_tuple
        fom_sims_all = self.simulate_curves_fom(assets, nb_simulations, tenors_list=tenors_chosen, seed=set_seed)
        self.gen_days_number(asset)  # TODO: FIX THIS HERE!!!
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

    def simulate_spot_blocks_from_fom( self
                                     , fom_sims_all
                                     , asset_nb
                                     , m
                                     , nb_simulations
                                     , days_partition
                                     , hours_partition
                                     , days_tuple
                                     , tenors_chosen = None
                                     , set_seed      = None
                                     , cuda_ind      = False ):
        """
        Generates spot blocks of month m from fom_sims_all (used for a tolling model)

        :param m: month to simulate spot block from
        :type m: int
        :param days_partition: partition of a week, i.e. [[0, 1, 2, 3, 4], [5, 6]]
        :type days_partition: list[list[int]]
        :param hours_partition: hours for blocks [[6, 18], [12, 12]]
        :type hours_partition: list[list[int]]
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
            spot_sims = cuda_ops.vtpv(fom_sims, col_vec, tm_ind='t', transpose_ind=True).transpose()

        else:  # no cuda
            w_days = np.cumsum(np.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l],
                               axis=1)
            # fom in column format
            fom_sims = fom_sims_used[m, :].reshape((len(fom_sims_used[tenors_chosen.index(m), :]), 1))
            spot_sims = np.transpose(fom_sims * np.exp(-0.5 * cv_m**2 * days + cv_m * w_days))

        return spot_sims
