# Tolling model
import config

from typing import List, Tuple, Union, Dict

import datetime
import numpy as np
import mrds
import ds

from tolling.opd              import opd_1fuel, opd_1fuel_cu
from tolling.com_skew_tolling import ComSkewTolling
from forward_curve            import FwdCurve
from vols.vols                import Volatility

if config.CUDA_PRESENT:
    import pycuda.gpuarray as gpa
    import cuda.cuda_ops   as cuda_ops


class TollingModel(ComSkewTolling):
    """ Tolling dispatch model.
    """

    def __init__(self
                 , mkt_date   : datetime.date
                 , fwd_curves : List[FwdCurve]
                 , vol_curves : List[Volatility]
                 , days_partition  : List[List[int]]
                 , hours_partition : List[List[int]]
                 , toll_start      : datetime.date
                 , toll_end        : datetime.date
                 , power_blocks_names
                 , fuel_idx_name
                 , days_partition_names
                 , hours_partition_names
                 , cash_vols
                 , tolling_params
                 , discount_curve = None
                 , calc_date      = None  # datetime.date format
                 , adj_fwd_tenors_days    = None
                 , adj_vol_tenors_days    = None
                 , cash_fwd_tenors_days   = None
                 , cash_vol_tenors_days   = None
                 , manual_adj             = None
                 , cash_corr_adj          = None
                 , cuda_ind               = False ):
        """ Path per path tolling model. The optimal bounday is a function of the shadow costs
            and other parameters.

        :param calc_date: calculation date of the tolling model.
        :param toll_start: Start of the tolling deal.
        :param toll_end: End of the tolling deal
        :param tolling_params: parameters of the tolling dispatch. This should have the
                                 following fields:  they are of different types
                                   hrAtMax,
                                   hrAtMin,
                                   maxCap,
                                   minDisp,
                                   startFuel,
                                   startFuelCold,
                                   addFuelCost,
                                   VC,
                                   rampRate,
                                   shutdownSPin,
                                   minDownTime,
                                   minRunTime,
                                   fixedStartupCost,
                                   fixedStartupCostCold,
                                   maxMonthlyStarts,
                                   coldStartup,
                                   startupHorizon,
                                   shutdownHorizon,
                                   rampUpSPin,
                                   rampDownSPin,
                                   rampUpCost,
                                   rampDownCost,
                                   rampUpHorizon,
                                   rampDownHorizon
        """

        super().__init__( mkt_date
                        , fwd_curves= )

        self.tolling_params      = tolling_params
        self.nb_days     = tolling_params.nb_days
        self.mkt_date     = mkt_date
        self.calc_date    = calc_date
        self.toll_start  = toll_start
        self.toll_end    = toll_end
        self.power_blocks_names = power_blocks_names

        self.cuda_ind = cuda_ind

        # these things are superflous
        self.adj_fwd_tenors_days = adj_fwd_tenors_days
        self.adj_vol_tenors_days = adj_vol_tenors_days
        self.cash_fwd_tenors_days = cash_fwd_tenors_days
        self.cash_vol_tenors_days = cash_vol_tenors_days
        self.manual_adj = manual_adj
        self.cash_corr_adj = cash_corr_adj

        self.fuel_idx_name = fuel_idx_name
        self.days_partition = days_partition
        self.days_partition_names = days_partition_names
        self.hours_partition = hours_partition
        self.hours_partition_names = hours_partition_names
        self.cash_vols = cash_vols

        fixed_monthly_val = None if self.fuel_idx_name is not 'FIXED' else self.tolling_params['fixedCostPerMonth']

        # tolling support vectors
        self.days_toll, self.days_d_toll, self.days_diff_toll, self.days_diff_l_toll = \
            self.power_models._generate_days_vecs(self.hours_partition, self.days_partition)

        # for usage w/ this class
        self.__power_models     = None
        self.__power_gas_blocks = None

    @classmethod
    def from_db(cls, mkt_date : datetime.date, block_names : List[List[str]]):
        """ Initializes the tolling model by reading the parameters from database.
        """

        # extracts all the commodities from the blocks.
        all_coms = set()
        for day_block in block_names:
            for hour_block in day_block:
                all_coms.add(hour_block)

        return cls(mkt_date)

    @property
    def power_gas_blocks(self) -> List[str]:
        """ Constructs the power and gas blocks used in the model.
        """

        if self.__power_gas_blocks:
            return self.__power_gas_blocks

        power_gas_blocks = set([item
                                for sublist in self.power_blocks_names
                               for item in sublist])
        power_gas_blocks.add(self.fuel_idx_name)

        self.__power_gas_blocks = list(power_gas_blocks)
        return self.__power_gas_blocks

    @staticmethod
    def __belongs_to_group(day_week : int, days_partition : List[List[int]]):
        """ Checks if the day in the week
        TODO: ELABORATE BETTER WHAT THIS IS DOING

        :param day_week: day in the week
        :param days_partition: partition of the week, e.g.  [[0,1,2,3,4],[5,6]]
        """
        idx_nb = 0
        for k in days_partition:
            if day_week in k:
                return idx_nb
            else:
                idx_nb += 1

    @staticmethod
    def construct_consequitive_hours( days_partition        : List[List[int]]
                                    , hours_partition       : List[List[int]]
                                    , nb_days               : int
                                    , block_names_partition : List[List[int]] ) -> Tuple[np.array, List]:
        """ Construct consequitive hours for the tolling model, used for spot simulation process.

        :param days_partition:  partition list [[0,1,2,3,4],[5,6]]
        :param hours_partition: hours list by blocks [[8, 16], [12,12]]
        :type nb_days: number of days
        :param block_names_partition: same as days_partition just for blocks
        """

        blocks_seq = []
        blocks_name_seq = []
        for day in range(nb_days):
            day_week = np.mod(day, 7)
            block_group_idx = TollingModel.__belongs_to_group(day_week, days_partition)
            blocks_seq += hours_partition[block_group_idx]
            blocks_name_seq += block_names_partition[block_group_idx]

        return np.array(blocks_seq, dtype=np.int32), blocks_name_seq

    def _power_fuel_process(self
                            , nb_sim             : int
                            , power_blocks_names : List[List[str]]
                            , fuel_idx_name      : str
                            , nb_days            : int
                            , fixed_monthly        = None
                            , cash_vols_overwrite  = False
                            , cash_fwd_tenors_days = None
                            , cash_vol_tenors_days = None
                            , manual_adj           = None):
        """ Simulate spot prices in the tolling model, by blocks.

        :param nb_sim: nb_sims. simulations
        :param days_partition: partition of the week, e.g. [[0,1,2,3,4],[5,6]]
        :param days_parition_names: partition names ['weekday', 'weekend']
        :param power_block_names: power blocks
            as in [ ['ERCOT_NORTH-PEAK'   , 'ERCOT_NORTH-OFFPEAK']
                  , ['ERCOT_NORTH-OFFPEAK', 'ERCOT_NORTH-OFFPEAK']]
        :param hours_partition: in form [[6,18],[12,12]]
        :param hours_partition_names: in form [['peak', 'offpeak'],['offpeak', 'offpeak']]
        :param fuel_idx_name: name of the fuel  TODO: THIS SHOULD BE REFACTORED.
        :param cash_vols: cash volatilities
        :param nb_days: number of days (TODO: WHAT DAYS!!! )
        """

        # obtaining the months for calibration
        toll_end_month = np.sum([ft < self.toll_end
                                 for ft in ds.get_forward_curve(power_blocks_names[0][0], self.calc_date)[0]])
        nb_fwds = toll_end_month

        power_gas_blocks = set([item
                                for sublist in power_blocks_names
                                for item in sublist])  # different blocks
        power_gas_blocks.add(fuel_idx_name)
        # mapping from names to number, works as: power_gas_blocks['ATSI_7X8'] gives 1 e.g.
        power_gas_block_idx = {pg_name: pg_idx for pg_name, pg_idx in
                               zip(power_gas_blocks, range(len(power_gas_blocks)))}

        power_gas_cash_vol_names = set([item
                                        for sublist in power_blocks_names
                                        for item in sublist])  # different blocks
        power_gas_cash_vol_names.add(fuel_idx_name)

        for days_bl, days_cv, fuel_cv in zip(power_blocks_names,
                                             cash_vols['power'],
                                             cash_vols['fuel']):
            for hour_bl, hour_cv, hour_fuel_cv in zip(days_bl, days_cv, fuel_cv):
                pg_idx = power_gas_block_idx[hour_bl]
                if not cash_vols_overwrite:
                    adj_fwd_tenors, adj_vol_tenors = mrds.find_adj_tenors(pg_idx, cash_fwd_tenors_days, cash_vol_tenors_days)
                    _, fwd_vol_values_unexpired, _, fwd_vol_tenors = ds.get_fwd_vol_curve_numeric_tenor( hour_cv
                                                                       , self.mkt_date
                                                                       , fwd_vol_ind         = 'vol'
                                                                       , adj_tenors_days = adj_vol_tenors )

                    cash_vol_write = np.array(fwd_vol_values_unexpired[:(nb_fwds + 1)], dtype=np.double)
                    self.power_models.set_cash_vols(pg_idx, cash_vol_write)
                else:
                    self.power_models.set_cash_vols(pg_idx, cash_vols)

        fuel_com_nb = self.power_models.nb_assets - 1
        adj_fwd_tenors, adj_vol_tenors = mrds.find_adj_tenors(fuel_com_nb, cash_fwd_tenors_days, cash_vol_tenors_days)
        _, fwd_vol_values_unexpired, _, fwd_vol_tenors = ds.get_fwd_vol_curve_numeric_tenor( cash_vols['fuel'][0][0]
                                                                                           , self.mkt_date
                                                                                           , fwd_vol_ind     = 'vol'
                                                                                           , adj_tenors_days = adj_vol_tenors )
        cash_vols_fuel = np.array(fwd_vol_values_unexpired[:(nb_fwds + 1)], dtype=np.double)

        # setting of cash vols
        self.power_models.set_cash_vols(power_gas_block_idx[fuel_idx_name], cash_vols_fuel)

        if manual_adj is not None:  # real HARD override
            exec(manual_adj)

        return self._power_fuel_process_reduced(nb_sim, power_gas_block_idx), power_gas_block_idx

    def _power_fuel_process_reduced(self, nb_sim  : int, power_gas_block_idx):
        """
        TODO: COMPLETELY REFACTOR THIS!!!

        """

        fwd_tenors_dt = self.power_models.forward_tenors_dt_list[0]
        tenors_chosen    = range( max(np.sum([ft < self.toll_start for ft in fwd_tenors_dt]) - 1, 0)
                                , max(np.sum([ft < self.toll_end   for ft in fwd_tenors_dt]) - 1, 0) + 1)

        fom_sims_all = [self.power_models.simulate_1nb(asset, nb_sim, tenors_chosen)  # tenors_chosen are simulation times TODO: CHECK THIS HERE
                        for asset in self.power_models.nb_assets]

        power_fuel_foms = [[(fom_sims_all[power_gas_block_idx[mo]],
                             fom_sims_all[power_gas_block_idx[self.fuel_idx_name]])
                            for mo in model_block_l]
                           for model_block_l in self.power_blocks_names]

        return { 'tenors_chosen'  : tenors_chosen
               , 'fom_sims_all'   : fom_sims_all
               , 'power_fuel_foms': power_fuel_foms }

    @property
    def power_models(self):
        """ Returns the power models applicable to this tolling model.
        """

        if self.__power_models:
            return self.__power_models

        self.__power_models = ComSkewTolling.from_db(self.calc_date, self.power_gas_blocks)

        return self.__power_models

    @property
    def _ndarray_type(self):
        return np.empty if not self.cuda_ind else gpa.empty

    @property
    def dispatch_mode(self) -> str:
        """ In which dispatch mode you were
        """

        return 'cmg'

    def _generate_spots(self, month_date : datetime.date, nb_sim : int):
        """ Generates power and fuel spots from month month_date.

        :param month_date: date designating the month of for tolling. Only year and month is used.
        :param nb_sim: number of simulations.
        """

        spot_blocks_m = [self.power_models.simulate_spot_blocks_from_fom( self.fom_sims_all
                                                                        , asset_nb
                                                                        , month_date.month
                                                                        , nb_sim
                                                                        , self.days_partition
                                                                        , self.hours_partition
                                                                        , (self.days_toll, self.days_d_toll, self.days_diff_toll, self.days_diff_l_toll)
                                                                        , tenors_chosen = self.tenors_chosen
                                                                        , cuda_ind      = self.cuda_ind )
                         for asset_nb in range(self.power_models.nb_assets)]

        # power fuel spots for month m
        pf_spots_m = [[(spot_blocks_m[self.power_gas_block_idx[mo]],
                        spot_blocks_m[self.power_gas_block_idx[self.fuel_idx_name]])
                       for mo in model_block_l]
                      for model_block_l in self.power_blocks_names]

        total_nb_blocks = 0
        for day in range(self.nb_days):
            day_week = np.mod(day, 7)
            for dp, psim in zip(self.days_partition, pf_spots_m):
                if day_week in dp:
                    total_nb_blocks += len(psim)

        power_sims = self._ndarray_type((total_nb_blocks, nb_sim))
        fuel_sims  = self._ndarray_type((total_nb_blocks, nb_sim))

        block_count = 0
        for day in range(self.nb_days):
            day_week = np.mod(day, 7)
            for dp, psim in zip(self.days_partition, pf_spots_m):
                if day_week in dp:
                    for ms, fs in psim:
                        power_sims[block_count, :] = ms[day, :]
                        fuel_sims[block_count, :] = fs[day, :] if self.fuel_idx_name != 'FIXED' else 1.  # TODO: NEEDS A FIX
                        block_count += 1

        return power_sims, fuel_sims

    def _start_shut_dispatch(self, cs):
        """ Dispatch decision according to shut hours, total starts.

        :param cs: current state vector of the power plant.
        """

        if not self.cuda_ind:
            cs['can_start'] = (cs.total_starts < self.tolling_params['maxMonthlyStarts']) & \
                              (cs.hours_shut >= self.tolling_params['minDownTime'])
        else:  # this kernel implements exactly what is above 3 lines
            cs['can_start'] = cuda_ops.comp_two_arrays_and(cs.total_starts
                                                           , cs.hours_shut
                                                           , self.tolling_params['maxMonthlyStarts']
                                                           , self.tolling_params['minDownTime'])

    def peak_only_startup(self, cs):
        """ Startup only at peak times.

        :param cs: current state vector
        """

        cnd_1 = cs['total_starts'] < self.tolling_params['maxMonthlyStarts']
        cnd_2 = cs['block_name'] == 'peak'
        cnd_3 = cs['hours_shut'] >= self.tolling_params['minDownTime']

        cs['can_start'] = cnd_1 & cnd_2 & cnd_3 if not self.cuda_ind else cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def _offpeak_only_startup(self, cs):
        """ Startup only at peak times.
        """

        cnd_1 = cs.total_starts < self.tolling_params['maxMonthlyStarts']
        cnd_2 = cs.block_name != 'peak'
        cnd_3 = cs.hours_shut >= self.tolling_params['minDownTime']

        cs['can_start'] = cnd_1 & cnd_2 & cnd_3 if not self.cuda_ind else cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def _startup_decision(self, cs):
        """ Decision whether to start up.

        :param cs: current state
        """

        startup = { 'cmg'         : self._start_shut_dispatch
                  , 'peak_only'   : self.peak_only_startup
                  , 'offpeak_only': self._offpeak_only_startup}

        startup[self.dispatch_mode](cs, self.tolling_params)

    def _forced_startup(self
                        , cs
                        , power_prices
                        , fuel_prices
                        , dv            = None):
        """ Try to force power plant to start.

        :param cs: current state object
        :param power_prices: vector of power prices
        :param fuel_prices: vector of fuel prices
        """

        dispatch_mode = self.dispatch_mode

        if dispatch_mode == 'mrg':
            decision_1 = power_prices - self.tolling_params['hrAtMax'] * fuel_prices
            cs['force_start'] = 2 * (decision_1 > dv[1]) + (decision_1 > dv[0]) & (decision_1 < dv[1])
        elif dispatch_mode == 'peak_only':
            cs['force_start'].fill(2 * (cs['block_name'] == 'peak'))
        elif dispatch_mode == 'offpeak_only':
            cs['force_start'].fill(2 * (cs['block_name'] != 'peak'))

    def _shut_start_dispatch(self, cs : Dict) -> None:
        """ Indicator function if the power plant can shut down. The condition is if hours_run is bigger
            than the minimum runtime. Replaces the 'can_shut' part of the current state dictionary with the new state.
        """

        if not self.cuda_ind:
            cs['can_shut'] = cs['hours_run'] >= self.tolling_params['minRunTime']
        else:
            cs['can_shut'] = cuda_ops.comp_array_number(cs['hours_run'], self.tolling_params['minRunTime'], op='larger', dtype='int32')

    def _peak_only_shutdown(self, cs : Dict) -> None:
        """

        """

        cnd_2 = cs['block_name'] != 'peak'
        cnd_3 = cs['hours_run'] >= self.tolling_params['minRunTime']

        if not self.cuda_ind:
            cs.can_shut = cnd_2 & cnd_3
        else:
            cuda_ops.min_int_two(cnd_2, cnd_3.astype(np.int32), cs.can_shut)

    def _offpeak_only_shutdown(self, cs : Dict) -> None:

        cnd_2 = cs['block_name'] == 'peak'
        cnd_3 = cs['hours_run']  >= self.tolling_params['minRunTime']

        if not self.cuda_ind:
            cs['can_shut'] = cnd_2 & cnd_3
        else:
            cuda_ops.min_int_two(cnd_2, cnd_3, cs['can_shut'])

    def _shutdown_decision(self, cs : Dict) -> None:
        """ Decision whether it is sensible to shut down

        :param cs: current state, cs.can_shut is filled by this routine
           cs.can_shut is array of bools

        """

        { 'cmg'         : self._shut_start_dispatch
        , 'peak_only'   : self._peak_only_shutdown
        , 'offpeak_only': self._offpeak_only_shutdown}[self.dispatch_mode](cs, self.tolling_params)

    def _forced_shutdown(self
                         , cs : Dict
                         , power_prices : Union[np.ndarray, gpa.GPUArray]
                         , fuel_prices  : Union[np.ndarray, gpa.GPUArray]
                         , dv = None):
        """ Decision to forcefully shut down, can take 3 outcomes: 2, 1, 0

        """

        dispatch_mode = self.dispatch_mode

        if dispatch_mode == 'mrg':
            decision_1 = power_prices - self.tolling_params['hrAtMax'] * fuel_prices
            cs['force_shut'] = 2 * (decision_1 < dv[3]) + (decision_1 > dv[2]) & (decision_1 < dv[3])
        elif dispatch_mode == 'peak_only':
            cs['force_shut'].fill(2 * (cs['block_name'] != 'peak'))
        elif dispatch_mode == 'offpeak_only':
            cs['force_shut'].fill(2 * (cs['block_name'] != 'offpeak'))

    def _set_other_params(self, cs : Dict, pl : int):
        """ Set up the cs.force_ and cs.can_ parameters

        :param cs: current state vector
        :param pl: number of simulations for the month

        """

        dispatch_mode = self.dispatch_mode

        if dispatch_mode == 'cmg' or dispatch_mode == 'mrg':
            if not self.cuda_ind:
                cs.force_start = np.ones(pl, dtype=np.short)
                cs.force_shut  = np.ones(pl, dtype=np.short)
                cs.can_start   = np.empty(pl, dtype=np.short)  # CHECK THIS ONE
                cs.can_shut    = np.empty(pl, dtype=np.short)
            else:  # cuda
                cs.force_start = gpa.empty(pl, dtype=np.int32).fill(1)
                cs.force_shut  = gpa.empty(pl, dtype=np.int32).fill(1)
                cs.can_start   = gpa.empty(pl, dtype=bool)
                cs.can_shut    = gpa.empty(pl, dtype=bool)

        elif dispatch_mode == 'always_run':
            if not self.cuda_ind:  # cpu
                cs.force_start = np.empty(pl, dtype=np.short).fill(2)
                cs.force_shut  = np.zeros(pl, dtype=np.short)
                cs.can_start   = np.empty(pl, dtype=np.short).fill(1)
                cs.can_shut    = np.empty(pl, dtype=np.short).fill(0)
            else:  # cuda
                cs.force_shut  = gpa.empty(dtype=np.int32).fill(0)  # force shut done once
                cs.force_start = gpa.empty(dtype=np.int32).fill(2)  # force start set here
                cs.can_shut    = gpa.zeros(pl, dtype=bool)
                cs.can_start   = gpa.zeros(pl, dtype=bool) + 1

        else:  # peak & offpeak only
            if not self.cuda_ind:  # cpu
                cs.force_start = np.empty(pl, dtype=np.short)
                cs.force_shut  = np.empty(pl, dtype=np.short)
                cs.can_start   = np.empty(pl, dtype=np.short)
                cs.can_shut    = np.empty(pl, dtype=np.short)
            else:  # cuda
                cs.force_start = 1  # TODO: FIX FIX FIX FIX
                cs.force_shut  = 1
                cs.can_start   = 1
                cs.can_shut    = 1

    def _block_dispatch(self
                        , spot_idx
                        , power_prices : Union[np.ndarray, gpa.GPUArray]
                        , fuel_prices  : Union[np.ndarray, gpa.GPUArray]
                        , startup_shadow_price
                        , cs      : Dict
                        , nb_sims : int
                        , opd_dispatch_fct):
        """ Dispatch in a single block, changes the current state as appropriate.

        :param spot_idx:
        :param power_prices: vector of power prices
        :param fuel_prices:
        :param startup_shadow_price: vector of startup shadow prices.
        :param cs: current state, a dictionary of various elements
        :param nb_sims: number of simulations
        :param opd_dispatch_fct: function for one-period dispatch algorithm
        """

        opd_dispatch_fct(spot_idx
                         , power_prices
                         , fuel_prices
                         , self.tolling_params
                         , startup_shadow_price
                         , cs['state']
                         , cs['hours_in_state']
                         , cs['generation']
                         , cs['total_starts']
                         , cs['hours_shut']
                         , cs['hours_run']
                         , cs['global_starts']
                         , cs['can_start']
                         , cs['can_shut']
                         , cs['force_start']
                         , cs['force_shut']
                         , cs['hours_block']
                         , cs['df']
                         , nb_sims
                         , cf_per_path_tmp
                         , cs)

    def dispatch_month( self
                      , m : int
                      , conseq_hours
                      , conseq_block_names
                      , pl
                      , power_spots
                      , fuel_spots
                      , dv            = None):
        """ Calculate the dispatch for month m

        :param m: month number
        :param dv: decision variable, for optimization
        """

        np_gpa = gpa if self.cuda_ind else np

        cf_per_path_tmp    = np_gpa.empty(pl)  # cash flow per path
        cf_per_path        = np_gpa.zeros(pl)  # cumulative cash flows

        cs = {'dispatch_mode':  self.dispatch_mode}
        self._set_other_params(cs, pl)
        cs['df'] = 1.

        # one period dispatch function
        opd_f = opd_1fuel.opd_1fuel if not self.cuda_ind else opd_1fuel_cu.opd_kernel

        for spot_idx, (block_hours, block_name) in enumerate(zip(conseq_hours, conseq_block_names)):
            power_prices, fuel_prices = power_spots[spot_idx, :], fuel_spots[spot_idx, :]

            # new state update
            cs.update( { 'hours_block'   : block_hours
                       , 'block_name'    : block_name } )

            if self.dispatch_mode == 'cmg':
                start_cost_init = self.tolling_params['fixedStartupCostCold'] + \
                                  self.tolling_params['SF'] * (fuel_prices + self.tolling_params['addFuelCost']) - \
                                  self.tolling_params['E_S'] * power_prices
                # startup shadow prices
                startup_sp_in = start_cost_init / (self.tolling_params['maxCap'] * self.tolling_params['startupHorizon'])
            else:
                startup_sp_in = 0.

            self.tolling_params['startup_shadow_price'] = startup_sp_in

            # the reason why first are overwritten is that they change very often
            self._startup_decision (cs)
            self._shutdown_decision(cs)

            # the following updates cs.force_shut, cs.force_start
            self._forced_startup (cs, power_prices, fuel_prices, dv)
            self._forced_shutdown(cs, power_prices, fuel_prices, dv)

            # dispatch for a block.
            self._block_dispatch(spot_idx
                                 , power_prices
                                 , fuel_prices
                                 , startup_sp_in
                                 , cs
                                 , len(power_prices)  # number of simulations
                                 , opd_f)

            cf_per_path += cf_per_path_tmp

        return self.power_models._discount_discount[m] * cf_per_path

    def dispatch(self, months_to_compute, nb_sim = 1000) -> Dict:
        """ Compute dispatch for all months in the model.

        :param months_to_compute: months for which to compute dispatch.
        :param nb_sim: number of simulations for the dispatch.
        """

        conseq_hours, conseq_block_names = TollingModel.construct_consequitive_hours( self.days_partition
                                                                                    , self.hours_partition
                                                                                    , self.nb_days
                                                                                    , self.hours_partition_names )

        dispatch_result = {}
        for month in months_to_compute:
            power_spots, fuel_spots = self._generate_spots(month)
            dispatch_result[month] = self.dispatch_month( month
                                                   , conseq_hours
                                                   , conseq_block_names
                                                   , nb_sim
                                                   , power_spots
                                                   , fuel_spots )

        if not self.cuda_ind:
            dispatch_res_months = {m: np.mean(dispatch_result[m]) for m in months_to_compute}
        else:
            dispatch_res_months = {m: gpa.sum(dispatch_result[m])/dispatch_result[m].size for m in months_to_compute}

        return { 'cashflow_by_month': dispatch_res_months
               , 'cashflow_total'   : sum(dispatch_res_months.values()) }
