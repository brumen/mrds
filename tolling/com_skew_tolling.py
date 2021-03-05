#   skew model used for tolling.
#

import datetime
import numpy as np

from logging  import getLogger
from typing   import List, Tuple, Dict, Union, Callable, Optional
from calendar import Calendar

from mrds.mrds_orm      import ComSkewORM
from mrds.vols.vols     import Volatility
from mrds.forward_curve import FwdCurve
from mrds.vols.vols_get import get_vol_object
from mrds.correlations  import corr_hyp_sec_two_fronts_time_diff
from cuda.cuda.cuda_ops import vtpm

from pycuda.gpuarray import GPUArray
from pycuda.compiler import SourceModule

# cuda related stuff
import pycuda.autoinit
import pycuda.curandom
import pycuda.cumath
import pycuda.gpuarray as gpa
import cuda.cublas.curand as curand
import skcuda.linalg
# from skcuda.cublas import cublasCreate, cublasSger, cublasDger

skcuda.linalg.init()  # this is necessary to work
# cublas_handle = cublasCreate()  # create a handle

logger = getLogger(__name__)


class ComSkewTolling(ComSkewORM):
    """ Adds the methods responsible only for tolling simulation, etc.
    """

    _CALENDAR     = Calendar()  # calendar object for generating days
    _SKEW_FCT_DIR = '/home/brumen'

    def __init__(self
                 , mkt_date        : datetime.date
                 , fwd_curves      : List[FwdCurve]
                 , vol_curves      : List[Volatility]
                 , cash_vol_curves : List[Volatility]
                 , cash_corrs      : Optional[Callable]
                 , days_partition  : Dict[str, Tuple]
                 , hours_partition : Dict[str, List[Tuple[str, int]]]
                 , discount_curve  : Optional[Callable] = None
                 , calc_date       : datetime.date      = None
                 , dcf             : float              = 365.25
                 , cuda_ind        : bool               = False ):

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
        :param cuda_ind: whether to use cuda, default False
        """

        super().__init__( mkt_date
                        , fwd_curves
                        , vol_curves
                        , discount_curve
                        , calc_date = calc_date
                        , dcf       = dcf
                        , cuda_ind  = cuda_ind )

        self.__cash_vol_curves   = cash_vol_curves
        self.__cash_correlations = cash_corrs
        self.days_partition     = days_partition
        self.hours_partition    = hours_partition

        # Cuda generator
        self.__gen = None

    @classmethod
    def from_db( cls
               , mkt_date        : datetime.date
               , fwd_curves      : List[str]
               , vol_curves      : List[str]
               , cash_vol_curves : List[str]
               , cash_corrs      = None
               , days_partition  = {'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,) }
               , hours_partition = { 'WEEKDAY': [('PJMW-PEAK', 8), ('PJMW-OFFPEAK', 16),]
                                   , 'WEEKEND': [('PJMW-PEAK', 16), ('PJMW-OFFPEAK', 8),] }
               , discount_curve  : Optional[Callable] = None
               , calc_date       : datetime.date      = None
               , dcf             : float              = 365.25
               , cuda_ind        : bool               = False  ):
        """ Obtains forward, vol curves from database.

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
        :param cuda_ind: whether to use cuda, default False
        """

        return cls( mkt_date
                  , [FwdCurve.from_db(mkt_date, fwd_curve) for fwd_curve in fwd_curves]
                  , [get_vol_object(fwd_curve, mkt_date)   for fwd_curve in fwd_curves]
                  , [get_vol_object(cash_vol_curve, mkt_date) for cash_vol_curve in cash_vol_curves]
                  , cash_corrs
                  , days_partition  = days_partition
                  , hours_partition = hours_partition
                  , discount_curve  = discount_curve
                  , calc_date       = calc_date
                  , dcf             = dcf
                  , cuda_ind        = cuda_ind )

    def _cash_vol_curves(self, asset : Union[str, List[str]]) -> Union[Volatility, Dict[str, Volatility]]:
        """ Returns the cash vol curve for the particular asset. If you enter the
            wrong asset, None is returned.

        :param asset: asset cash vol to be returned, or a list of asset cash vols
        :returns: the Volatility subclass for that asset, or a dictionary where keys are assets, and
                  values are Volatilities for those assets.
        """

        assert isinstance(asset, str) or isinstance(asset, list), 'asset parameter is either a string, or a list of strings.'

        if isinstance(asset, str):
            for fwd_curve, cash_vol_curve in zip(self.fwd_curves, self.__cash_vol_curves):
                if fwd_curve.fwd_name == asset:
                    return cash_vol_curve

        # asset is List[str]
        cash_vol_dict = {}
        for fwd_curve, cash_vol_curve in zip(self.fwd_curves, self.__cash_vol_curves):
            if fwd_curve.fwd_name in asset:
                cash_vol_dict[fwd_curve.fwd_name] = cash_vol_curve

        return cash_vol_dict

    def _cash_correlation(self
                         , asset_1 : str
                         , asset_2 : str
                         , fwd_date_1 : Optional[datetime.date] = None
                         , fwd_date_2 : Optional[datetime.date] = None
                         , default_corr = 0.95 ):
        """ Cash correlations between asset_1 and asset_2.

        :param asset_1: first asset to get correlations. ('ERCOT-PEAK')
        :param asset_2: second asset to compute correlations. ('ERCOT-OFFPEAK')
        :param fwd_date_1: date on the first curve (asset_1), possibly not needed,
        :param fwd_date_2: date on the second curve, possibly not needed
        :param default_corr: default correlation between asset_1 & asset_2, in hyp_sec form
        """

        if self.__cash_correlations:  # cash correlation is a given function
            return self.__cash_correlations(asset_1, asset_2, fwd_date_1, fwd_date_2)

        if fwd_date_1 is None or fwd_date_2 is None:  # ignoring the forward dates
            if asset_1 == asset_2:
                return 1.

            return default_corr

        # else: default correlation from the hyp-sec correlation structure.
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

    @property
    def __generator(self):
        """ Selects the random variable generator.

        """

        if self.__gen:
            return self.__gen

        self.__gen = pycuda.curandom.XORWOWRandomNumberGenerator()
        return self.__gen

    def _cash_rns(self, cash_corr : np.ndarray, nb_simulations : int ) -> Union[np.array, GPUArray]:
        """ Generates the cash correlations

        :param cash_corr: matrix of cash correlations
        :param nb_simulations: number of simulations.
        """

        nb_assets = cash_corr.shape[0]  # cash_corr is a square matrix

        if not self.cuda_ind:
            return np.random.multivariate_normal( np.zeros(nb_assets)
                                                , cash_corr
                                                , size = nb_simulations )

        # cuda requested.
        dtype_ = cash_corr.dtype

        if cash_corr.shape == (1, 1):
            cash_rns = gpa.empty(nb_simulations, dtype=dtype_)
            self.__generator.fill_normal(cash_rns)  # fills it w/ normal
            return cash_rns

        # multiply by cholesky
        cash_rns = gpa.empty((nb_assets, nb_simulations), dtype=dtype_)
        self.__generator.fill_normal(cash_rns)
        cash_corr_gpu = gpa.to_gpu(np.linalg.cholesky(cash_corr).astype(dtype_))
        return skcuda.linalg.dot(cash_corr_gpu, cash_rns).transpose()  # performs the matrix multiplication

    @staticmethod
    def _create_first_of_months( start_date : datetime.date
                               , end_date   : datetime.date) -> List[datetime.date]:
        """ Constructs a list of first of months between start_date and end_date (including the month where
            start_date is.

        :param start_date: start date for the month range
        :param end_date: end date of the month range.
        :returns: list of months between start and end date.
        """

        first_of_month = datetime.date(start_date.year, start_date.month, 1)

        if start_date.month == 12:
            next_first_of_month = datetime.date(start_date.year, 1, 1)
        else:
            next_first_of_month = datetime.date(start_date.year, start_date.month+1, 1)

        last_first_of_month = datetime.date(end_date.year, end_date.month, 1)

        # iterate between the next_first_of_month and last_first_of_month
        curr_month       = next_first_of_month
        curr_list_months = [first_of_month]

        while curr_month <= last_first_of_month:
            curr_list_months.append(curr_month)
            if curr_month.month != 12:
                curr_month = datetime.date(curr_month.year, curr_month.month+1, 1)
            else:
                curr_month = datetime.date(curr_month.year+1, 1, 1)

        return curr_list_months

    def __find_day_partition(self, day_nb : int) -> str:
        """ Finds the day_nb in the partition.

        :param day_nb: day enumerated to be found in the partition.
        :returns: partition name where the day is found.
        """

        for partition_name, partition in self.days_partition.items():
            if day_nb in partition:
                return partition_name

        raise RuntimeError(f'Couldnt find {day_nb} in partition {self.days_partition}')

    def _generate_hours(self, start_day_in_month : datetime.date, hours_partition=None) -> List[Tuple[str, int]]:
        """ Generate consequent hours for particular year and month.

        :param start_day_in_month: start day in the month for which hours are generated.
        :param hours_partition: override w/ new hours partition.
        :returns: list of tuples, where the first element of the tuple is a curve, and the second
                  the amount of hours in that curve.
        """

        hours_partition_used = hours_partition if hours_partition is not None else self.hours_partition

        hours = []
        start_year, start_month = start_day_in_month.year, start_day_in_month.month
        for year_, month_, day_, day_in_week in self._CALENDAR.itermonthdays4(start_year, start_month):
            if year_ == start_year and month_ == start_month and day_ >= start_day_in_month.day:  # iterator gives days for other months too, to complete the week
                hours.extend(hours_partition_used[self.__find_day_partition(day_in_week)])

        return hours

    def __sub_bcast(self, a : GPUArray, b : GPUArray) -> GPUArray:
        """ Subtracts along the broadcasted edge like in numpy
        IMPORTANT: THIS COULD BE REALLY BAD IN TERMS OF EFFICIENCY

        :param a: first array
        :param b: second array
        :returns: a-b, including
        """

        import tensorflow as tf

        with tf.device('/GPU:0'):
            return tf.subtract(tf.constant(a), tf.constant(b))

    def simulate_spot_blocks( self
                            , assets         : List[str]
                            , nb_simulations : int
                            , tolling_start  : datetime.date
                            , tolling_end    : datetime.date
                            , set_seed      = None
                            , hours_partition = None
                            , ignore_block_names : bool = False  ) -> Dict[datetime.date, List[Tuple[str, int, np.array]]]:
        """ Same as simulate_spot_blocks, but for all blocks. This is a GENERATOR.

        :param assets: list of assets to which asset to simulate block prices for.
        :param nb_simulations: number of simulations.
        :param tolling_start: start of the tolling simulations
        :param tolling_end: end of tolling sims.
        :param set_seed: optional param for debugging, so that simulations are always the same
        :param hours_partition: override for the hours partition (used for fuel stuff)
        :param ignore_block_names: ignore block names (USED ONLY FOR FUEL SIMULATION - FIX THIS)
        :returns: dictionary, where keys are simulated first-of-months, and values are:
                    tuples, where the first is block name, second is block hours, and third is the simulations for that
                            block.
        """

        # fom_sims type is: {date: {asset: sims}}
        fom_sims = self.simulate_1nb( assets
                                    , nb_simulations
                                    , self.__class__._create_first_of_months(tolling_start, tolling_end)
                                    , set_seed = set_seed )

        #
        # spot_sims = {}
        for sim_date, sim_info in fom_sims.items():  # sim_info = {'asset': simulations}
            all_assets = list(sim_info.keys())  # keys is a generator

            # correlations
            cash_corr_mtx = np.array([ [ self._cash_correlation(asset_1, asset_2)
                                    for asset_2 in all_assets ]
                                  for asset_1 in all_assets ])

            days_within_month_sim = self._generate_hours(sim_date, hours_partition=hours_partition)
            cash_vols = np.array([ self._cash_vol_curves(asset).atm_vol(sim_date) for asset in all_assets] )  # TODO: this is the same for all assets, OPTIMIZE

            first_block_name, first_block_hour = days_within_month_sim[0]
            spot_sim = [ ( first_block_name
                         , first_block_hour
                         , sim_info[first_block_name][0] if not self.cuda_ind else gpa.to_gpu(sim_info[first_block_name][0]) ) ]  # TODO: THIS 0 HERE MIGHT BE WRONG
            if not self.cuda_ind:
                curr_asset_values = np.array([sim_info[asset][0] for asset in all_assets]).transpose()  # each asset sims in a row
            else:
                curr_asset_values = gpa.to_gpu(np.array([sim_info[asset][0].tolist() for asset in all_assets])).transpose()

            for block_name, block_hours in days_within_month_sim:
                block_hours_num = block_hours / self.dcf
                cash_corr_rns = self._cash_rns(cash_corr_mtx, nb_simulations).transpose()

                # this performs the kronecker product basically
                if not self.cuda_ind:
                    curr_asset_values *= np.exp(-0.5 * cash_vols ** 2 * block_hours_num + \
                                                np.sqrt(block_hours_num) * cash_vols * cash_corr_rns.transpose() )
                else:
                    if cash_vols.shape == (1, ):  # 1 asset generation
                        curr_asset_values *= pycuda.cumath.exp(-0.5 * cash_vols**2 * block_hours_num + \
                                                   np.sqrt(block_hours_num) * cash_vols * cash_corr_rns.transpose() )
                    else:
                        cash_vols_gpu = cash_vols if isinstance(cash_vols, GPUArray) else gpa.to_gpu(cash_vols)
                        curr_asset_values *= pycuda.cumath.exp(vtpm( -0.5 * block_hours_num * cash_vols_gpu**2
                                                       , np.sqrt(block_hours_num) * vtpm( cash_vols_gpu, cash_corr_rns.transpose(), tm_ind='t', new_mtx_gen=True)
                                                       , tm_ind      = 'p'
                                                       , new_mtx_gen = True ) )

                spot_sim.append( ( block_name
                                 , block_hours
                                 , curr_asset_values[:, all_assets.index(block_name)  ]) )  # if not ignore_block_names else 0

            yield sim_date, spot_sim
            # spot_sims[sim_date] = spot_sim

        # return spot_sims


class ComSkewTollingCuda(ComSkewTolling):
    """ Cuda version of skew tolling model.
    """

    def __F_skew_tsf_cuda(self):
        """

        """

        with open(self._SKEW_FCT_DIR + 'cuda/skew_tsf.c', 'r') as F_skew_el:
            return SourceModule(F_skew_el.read()).get_function('F_skew_tsf')

    @staticmethod
    def _cash_rns( cash_corr      : np.ndarray
                 , nb_simulations : int
                 , rn_type        = float ):
        """ Generates the cash correlations

        :param cash_corr: matrix of cash correlations
        :param nb_simulations: number of simulations.
        :param rn_type: type of random numbers to generate.
        :param
        """

        nb_assets = cash_corr.shape[0]  # cash_corr is a square matrix

        spot_rn_init  = gpa.empty( (nb_assets, nb_simulations), dtype=rn_type)
        cash_corr_gpu = gpa.to_gpu(np.linalg.cholesky(cash_corr).astype(rn_type))
        curand.gen_eff_dev_rns( spot_rn_init.size
                              , np.longlong(spot_rn_init.ptr)
                              , pycuda.curandom.XORWOWRandomNumberGenerator())

        return cuda_ops.matmul(cash_corr_gpu, spot_rn_init)

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

        return spot_sims
