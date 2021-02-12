#   skew model used for tolling.
#

import datetime
import numpy as np

from logging  import getLogger
from typing   import List, Tuple, Dict, Union, Callable
from calendar import Calendar

from mrds.mrds_spot     import ComSkewSpot
from mrds.vols.vols     import Volatility
from mrds.forward_curve import FwdCurve
from mrds.vols.vols_get import get_vol_object

from pycuda.gpuarray import GPUArray
from pycuda.compiler import SourceModule

import pycuda.curandom
import pycuda.cumath
import pycuda.gpuarray as gpa


logger = getLogger(__name__)


class ComSkewTolling(ComSkewSpot):
    """ Adds the methods responsible only for tolling simulation, etc.
    """

    _CALENDAR     = Calendar()  # calendar object for generating days
    _SKEW_FCT_DIR = '/home/brumen'

    def __init__(self
                 , mkt_date        : datetime.date
                 , fwd_curves      : List[FwdCurve]
                 , vol_curves      : List[Volatility]
                 , cash_vol_curves : List[Volatility]
                 , cash_corrs      : Callable
                 , days_partition  : Dict[str, Tuple]
                 , hours_partition : Dict[str, List[Tuple[str, int]]]
                 , discount_curve = None
                 , calc_date      = None
                 , dcf            : float = 365.25 ):

        """ Initialization of the skew model for tolling simulation.

        :param mkt_date: market date
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, same structure as fwd_curves, but the objects are volatility objects.
        :param cash_vol_curves: cash vol curves, same structure as the vol_curves.
        :param cash_corrs: cash correlations - double dictionary of numbers.
        :param discount_curve: discount curve, a function of fwd_date, returns lambda fwd_date: discount(mkt_date, fwd_date)
        :param calc_date: calculation date.
        :param days_partition: partition of days,  Mon = 0, Sun = 6, e.g. [[0,1,2,3,4], [5,6]]  # TODO: MAYBE CHANGE THIS
                               {'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,)
        :param hours_partition: partition of hours for each block, e.g { 'WEEKDAY': ((PJMW-PEAK, 8), (PJMW-OFFPEAK, 16),)
                                                                       , 'WEEKEND': ((PJMW-PEAK, 16), (PJMW-OFFPEAK, 8),) }
        :param dcf: day-count factor.
        """

        super().__init__( mkt_date
                        , fwd_curves
                        , vol_curves
                        , cash_vol_curves
                        , cash_correlations = cash_corrs
                        , discount_curve    = discount_curve
                        , calc_date         = calc_date
                        , dcf               = dcf )

        self.days_partition  = days_partition
        self.hours_partition = hours_partition

    # TODO: FIX THIS METHOD A BIT
    @classmethod
    def from_db( cls
               , mkt_date        : datetime.date
               , fwd_curves      : List[str]
               , vol_curves      : List[str]
               , cash_vol_curves : List[str]
               , cash_corrs = None
               , days_partition = {'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,) }
               , hours_partition = { 'WEEKDAY': [('PJMW-PEAK', 8), ('PJMW-OFFPEAK', 16),]
                                   , 'WEEKEND': [('PJMW-PEAK', 16), ('PJMW-OFFPEAK', 8),] } ):

        return cls( mkt_date
                  , [FwdCurve.from_db(mkt_date, fwd_curve) for fwd_curve in fwd_curves]
                  , [get_vol_object(fwd_curve, mkt_date)   for fwd_curve in fwd_curves]
                  , [get_vol_object(cash_vol_curve, mkt_date) for cash_vol_curve in cash_vol_curves]
                  , cash_corrs
                  , days_partition  = days_partition
                  , hours_partition = hours_partition )

    def __find_day_partition(self, day_nb : int) -> str:
        """ Finds the day_nb in the partition.

        :param day_nb: day enumerated to be found in the partition.
        :returns: partition name where the day is found.
        """

        for partition_name, partition in self.days_partition.items():
            if day_nb in partition:
                return partition_name

        raise RuntimeError(f'Couldnt find {day_nb} in partition {self.days_partition}')

    def _generate_hours(self, start_day_in_month : datetime.date) -> List[Tuple[str, int]]:
        """ Generate consequent hours for particular year and month.

        :param start_day_in_month: start day in the month for which hours are generated.
        :returns: list of tuples, where the first element of the tuple is a curve, and the second
                  the amount of hours in that curve.
        """

        hours = []
        start_year, start_month = start_day_in_month.year, start_day_in_month.month
        for year_, month_, day_, day_in_week in self._CALENDAR.itermonthdays4(start_year, start_month):
            if year_ == start_year and month_ == start_month and day_ >= start_day_in_month.day:  # iterator gives days for other months too, to complete the week
                hours.extend(self.hours_partition[self.__find_day_partition(day_in_week)])

        return hours

    def _generate_days_vecs(self, nb_days: int) -> Union[Tuple[List, List], Tuple[GPUArray, GPUArray]]:
        """ Generate days for simulate_spot_blocks.

        :param cuda_ind: Whether to use and generate objects on cuda, or on cpu.
        :returns: tuple of days TODO: FIX THIS
        """

        days = self._generate_days(nb_days)
        days_diff = np.empty(len(days))
        days_diff[0] = 0.
        days_diff[1:] = np.diff(days)

        return days, days_diff

    @staticmethod
    def __generate_fom(start_date : datetime.date, end_date: datetime.date) -> List[datetime.date]:
        """ Generate all first-of-month between start_date and end_date, including month w/ start_date.

        :param start_date: start date for FOM generation
        :param end_date: end date for FOM generation
        :returns:

        """

        foms = []

        if start_date.day != 1:
            foms.append(datetime.date(start_date.year, start_date.month, 1))

        # go through the months, add first of month
        start_year = start_date.year
        end_year   = end_date.year
        for year_ in range(start_year, end_year + 1):
            start_month = start_date.month if year_ == start_year else 1
            end_month   = end_date.month   if year_ == end_year   else 12
            for month_ in range(start_month, end_month + 1):
                foms.append(datetime.date(year_, month_, 1))

        return foms

    def simulate_spot_blocks( self
                            , assets         : List[str]
                            , nb_simulations : int
                            , tolling_start  : datetime.date
                            , tolling_end    : datetime.date
                            , set_seed      = None) -> Dict[str, np.ndarray]:
        """ Same as simulate_spot_blocks, but for all blocks. TODO: DESCRIBE THIS BETTER

        :param assets: list of assets to which asset to simulate block prices for.
        :param nb_simulations: number of simulations.
        :param tolling_start: start of the tolling simulations
        :param tolling_end: end of tolling sims.
        :param set_seed: optional param for debugging, so that simulations are always the same
        :returns: dictionary, where keys are simulated assets, and values are TODO: FINISH HERE!!!
        """

        # fom_sims type is: {date: {asset: sims}}
        fom_sims = self.simulate_1nb( assets
                                    , nb_simulations
                                    , self.__class__._create_first_of_months(tolling_start, tolling_end)
                                    , set_seed = set_seed )

        spot_sims = {}
        for sim_date, sim_info in fom_sims.items():  # sim_info = {'asset': simulations}
            all_assets = list(sim_info.keys())  # keys is a generator

            # correlations
            cash_corr_mtx = np.array([ [ self._cash_correlation(asset_1, asset_2)
                                    for asset_2 in all_assets ]
                                  for asset_1 in all_assets ])

            days_within_month_sim = self._generate_hours(sim_date)
            cash_vols = np.array([ self._cash_vol_curves(asset).atm_vol(sim_date) for asset in all_assets] )  # TODO: this is the same for all assets, OPTIMIZE

            first_block_name, first_block_hour = days_within_month_sim[0]
            spot_sim = [ (first_block_name, first_block_hour, sim_info[first_block_name][0]) ]  # TODO: THIS 0 HERE MIGHT BE WRONG
            curr_asset_values = np.array([sim_info[asset][0] for asset in all_assets]).transpose()  # each asset sims in a row

            for block_name, block_hours in days_within_month_sim:
                block_hours_num = block_hours / self.dcf
                cash_corr_rns = self.__class__._cash_rns(cash_corr_mtx, nb_simulations).transpose()
                curr_asset_values *= np.exp(-0.5 * cash_vols**2 * block_hours_num + \
                                            np.sqrt(block_hours_num) * cash_vols * cash_corr_rns.transpose())
                spot_sim.append( (block_name, block_hours, curr_asset_values[:, all_assets.index(block_name)]) )

            spot_sims[sim_date] = spot_sim

        return spot_sims


class ComSkewTollingCuda(ComSkewTolling):
    """ Cuda version of skew tolling model.
    """

    def __F_skew_tsf_cuda(self):
        """

        """

        with open(self._SKEW_FCT_DIR + 'cuda/skew_tsf.c', 'r') as F_skew_el:
            return SourceModule(F_skew_el.read()).get_function('F_skew_tsf')

    def _generate_days_vecs(self, nb_days: int, cuda_ind=False) -> Union[Tuple[List, List], Tuple[GPUArray, GPUArray]]:
        """ Generate days for simulate_spot_blocks.

        :param cuda_ind: Whether to use and generate objects on cuda, or on cpu.
        :returns: tuple of days TODO: FIX THIS
        """

        days = self._generate_days(nb_days)
        days_diff = gpa.empty(len(days))
        days_diff[0] = np.array(0.)
        days_diff[1:] = np.diff(days)  # TODO: PROBABLY THIS IS INEFFICIENT, CHECK!!

        return gpa.to_gpu(days), days_diff

    def simulate_spot_blocks( self
                            , assets         : List[str]
                            , nb_simulations : int
                            , tolling_start  : datetime.date
                            , tolling_end    : datetime.date
                            , set_seed       = None) -> Dict[str, np.ndarray]:
        """ Same as simulate_spot_blocks, but for all blocks. TODO: DESCRIBE THIS BETTER

        :param assets: list of assets to which asset to simulate block prices for.
        :param nb_simulations: number of simulations.
        :param tolling_start: start of the tolling simulations
        :param tolling_end: end of tolling sims.
        :param set_seed: optional param for debugging, so that simulations are always the same
        :returns: dictionary, where keys are simulated assets, and values are TODO: FINISH HERE!!!
        """

        # obtain the months corresponding to tolling_start and tolling_end.
        first_month = datetime.date(tolling_start.year, tolling_start.month, 1)  # first of first month
        last_month  = datetime.date(tolling_end.year  , tolling_end.month  , 1)  # first of last month
        months_to_use = 1  # TODO: dates of first of months between first_month and last_month
        days, days_diff = self._generate_days_vecs(nb_days)
        # construct the equiv. of days = range(31)/365.25
        fom_sims_fom = self.simulate_1nb(assets, nb_simulations, months_to_use, set_seed=set_seed)

        spot_sims = {}
        for asset in assets:

            self.gen_spot_rn(nb_simulations, cuda_ind=cuda_ind)

            cash_curves_asset = self._cash_vol_curves(asset)
            # cash vol tenors
            cv_tenors = [cash_curves_asset.implied_vol(fwd_date, K, ttm)
                         for fwd_date in (cash_curves_asset.vol_dates if not tenors_chosen else tenors_chosen)]

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
            # else:  # no cuda
            #     w_days = np.cumsum( np.sqrt(days_diff[:days_diff_l]) * self.spot_rn_a[asset_nb][:, :days_diff_l]
            #                       , axis=1)
            #     for fwd_tenor_nb, cash_vol_tenor in enumerate(cv_tenors):
            #         # fom in column format
            #         fom_sims_for_tenor = fom_sims_all[asset][fwd_tenor_nb, :]
            #         fom_sims = fom_sims_for_tenor.reshape((len(fom_sims_for_tenor), 1))  # column vector
            #         spot_sims[asset][fwd_tenor_nb] = np.transpose(fom_sims *
            #                                                np.exp(-0.5 * cash_vol_tenor**2 * days +
            #                                                       cash_vol_tenor * w_days))

        return spot_sims
