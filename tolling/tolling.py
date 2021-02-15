# Tolling model
import mrds.config

from typing import List, Tuple, Union, Dict, Callable, Optional, Any

import datetime
import numpy as np

from mrds.tolling.opd              import opd_1fuel, opd_1fuel_cu
from mrds.tolling.com_skew_tolling import ComSkewTolling
from mrds.tolling.default_tolling  import tolling_params_default
from mrds.forward_curve            import FwdCurve
from mrds.vols.vols                import Volatility
from mrds.vols.vols_get            import get_vol_object

import pycuda.gpuarray as gpa

if mrds.config.CUDA_PRESENT:
    import pycuda.autoinit  # leave this here to initialize the GPU
    import cuda.cuda_ops   as cuda_ops


class TollingModel(ComSkewTolling):
    """ Tolling dispatch model.
    """

    def __init__(self
                 , mkt_date        : datetime.date
                 , toll_start      : datetime.date
                 , toll_end        : datetime.date
                 , fwd_curves      : List[FwdCurve]
                 , vol_curves      : List[Volatility]
                 , days_partition  : Dict[str, Tuple]
                 , hours_partition : Dict[str, List[Tuple[str, int]]]
                 , fuel_idx        : str
                 , cash_vols       : List[Volatility]
                 , cash_corr       : Callable                 = None
                 , tolling_params  : Optional[Dict[str, Any]] = None
                 , discount_curve  : Callable                 = None
                 , calc_date       : Optional[datetime.date]  = None
                 , cuda_ind        : bool                     = False
                 , dcf             : float                    = 365.25 ):
        """ Path per path tolling model. The optimal boundary is a function of the shadow costs and other parameters.

        :param mkt_date: market date
        :param toll_start: start of the tolling deal.
        :param toll_end: end of the tolling deal
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, same structure as fwd_curves, but the objects are volatility objects.
        :param days_partition: partition of days,  Mon = 0, Sun = 6, e.g. [[0,1,2,3,4], [5,6]]  # TODO: MAYBE CHANGE THIS
                               {'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,)
        :param hours_partition: partition of hours for each block, e.g { 'WEEKDAY': ((PJMW-PEAK, 8), (PJMW-OFFPEAK, 16),)
                                                                       , 'WEEKEND': ((PJMW-PEAK, 16), (PJMW-OFFPEAK, 8),) }
        :param fuel_idx: name of the fuel to use.
        :param cash_vols: cash vol curves, same structure as the vol_curves.
        :param cash_corr: cash correlations - double dictionary of numbers.
        :param discount_curve: discount curve, a function of fwd_date, returns lambda fwd_date: discount(mkt_date, fwd_date)
        :param calc_date: calculation date.
        :param cuda_ind: indicator whether to use cuda
        :param dcf: day-count factor
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
                        , fwd_curves
                        , vol_curves
                        , cash_vols
                        , cash_corr
                        , days_partition
                        , hours_partition
                        , discount_curve = discount_curve
                        , calc_date      = calc_date
                        , dcf            = dcf )

        self.fuel_idx           = fuel_idx
        self.tolling_params     = tolling_params
        self.toll_start         = toll_start
        self.toll_end           = toll_end

        # some other parameters
        self.cuda_ind = cuda_ind

        # TODO: FIX THIS HERE.
        # fixed_monthly_val = None if self.fuel_idx_name is not 'FIXED' else self.tolling_params['fixedCostPerMonth']

        # for usage w/ this class
        self.__dispatch_mode    = 'cmg'  # default mode

    @classmethod
    def from_db( cls
               , mkt_date        : datetime.date
               , toll_start      : datetime.date
               , toll_end        : datetime.date
               , fwd_curves      : List[str]
               , vol_curves      : List[str]
               , days_partition  : Dict[str, Tuple]
               , hours_partition : Dict[str, List[Tuple[str, int]]]
               , fuel_idx        : str
               , cash_vol_curves : List[str]
               , cash_corrs      = None
               , tolling_params  : Optional[Dict[str, Any]] = None
               , discount_curve  : Optional[Callable]       = None
               , calc_date       : datetime.date            = None
               , dcf             : float                    = 365.25
               , cuda_ind        : bool                     = False ):
        """ Obtains forward, vol curves from database.

        :param mkt_date: market date
        :param toll_start: start of the tolling contract
        :param toll_end: end of the tolling contract.
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, same structure as fwd_curves, but the objects are volatility objects.
        :param cash_vol_curves: cash vol curves, same structure as the vol_curves.
        :param cash_corrs: cash correlations - double dictionary of numbers.
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
        :param discount_curve: discount curve, a function of fwd_date, returns lambda fwd_date: discount(mkt_date, fwd_date)
        :param calc_date: calculation date.
        :param days_partition: partition of days,  Mon = 0, Sun = 6, e.g. [[0,1,2,3,4], [5,6]]  # TODO: MAYBE CHANGE THIS
                               {'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,)
        :param hours_partition: partition of hours for each block, e.g { 'WEEKDAY': ((PJMW-PEAK, 8), (PJMW-OFFPEAK, 16),)
                                                                       , 'WEEKEND': ((PJMW-PEAK, 16), (PJMW-OFFPEAK, 8),) }
        :param fuel_idx: fuel index (e.g. 'NG..')
        :param dcf: day-count factor.
        :param cuda_ind: indicator of the cuda presence.
        """

        return cls( mkt_date
                  , toll_start
                  , toll_end
                  , [FwdCurve.from_db(mkt_date, fwd_curve) for fwd_curve in fwd_curves]
                  , [get_vol_object(fwd_curve, mkt_date)   for fwd_curve in fwd_curves]
                  , days_partition
                  , hours_partition
                  , fuel_idx
                  , [get_vol_object(vol_curve, mkt_date)   for vol_curve in cash_vol_curves]
                  , cash_corr      = cash_corrs
                  , tolling_params = tolling_params if tolling_params is not None else tolling_params_default()
                  , discount_curve = discount_curve
                  , cuda_ind       = cuda_ind
                  , dcf            = dcf )

    @property
    def dispatch_mode(self) -> str:
        """ In which dispatch mode you are.
        """

        return self.__dispatch_mode

    @dispatch_mode.setter
    def dispatch_mode(self, new_dispatch: str):

        assert self.__dispatch_mode in ('cmg', )

        self.__dispatch_mode = new_dispatch

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

    def dispatch_all( self
                      , tolling_start: datetime.date
                      , tolling_end  : datetime.date
                      , set_seed =None
                      , nb_simulations : int = 1000):
        """ Constructs the fuel process (self.fuel_idx) for the blocks.

        :param nb_simulations: number of simulations.
        :param tolling_start: start of the tolling simulations
        :param tolling_end: end of tolling sims.
        :param set_seed: optional param for debugging, so that simulations are always the same
        :returns: dictionary, where keys are simulated first-of-months, and values are:
                    tuples, where the first is block name (always fuel index), second is block hours, and third is the simulations for that block.
        """

        fuel_hours_partition = { weekday: tuple( (self.fuel_idx, nb_hours) for _, nb_hours in weekday_split )
            for weekday, weekday_split in self.hours_partition.items() }

        fuel_process = self.simulate_spot_blocks( [self.fuel_idx]
                                                , nb_simulations
                                                , tolling_start
                                                , tolling_end
                                                , set_seed = set_seed
                                                , hours_partition = fuel_hours_partition )  # replace old partition w/ fuel partition

        power_processes = self.simulate_spot_blocks( [fwd_curve.fwd_name for fwd_curve in self.fwd_curves]
                                                   , nb_simulations
                                                   , tolling_start
                                                   , tolling_end
                                                   , set_seed = set_seed )

        dates_months = list(fuel_process.keys())  # same as power_process.keys()

        dispatch_per_month = {}
        for date_month in dates_months:  # date_month - beginning of that month
            fuel_process_month  = fuel_process[date_month]  # (list of (block_name, block_hours, block_sims)
            power_process_month = power_processes[date_month]  # same here

            value_per_month = []
            for power_block, fuel_block in zip(power_process_month, fuel_process_month):
                power_block_name, block_hours, power_block_values = power_block
                fuel_block_name , _          , fuel_block_values  = fuel_block

                value_per_month.append( block_hours
                                      , (power_block_name, fuel_block_name)
                                      , self._dispatch_month(block_hours, (power_block_name, fuel_block_name), (power_block_values, fuel_block_values)) )

            dispatch_per_month[date_month] = value_per_month

        return dispatch_per_month

    def dispatch_month( self
                      , conseq_hours
                      , conseq_block_names
                      , pl
                      , power_spots
                      , fuel_spots
                      , dv            = None):
        """ Calculate the dispatch for month m

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
