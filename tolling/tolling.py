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

        # some other parameters
        self.cuda_ind = cuda_ind

        # TODO: FIX THIS HERE.
        # fixed_monthly_val = None if self.fuel_idx_name is not 'FIXED' else self.tolling_params['fixedCostPerMonth']

        # for usage w/ this class
        self.__dispatch_mode    = 'cmg'  # default mode

    @classmethod
    def from_db( cls
               , mkt_date        : datetime.date
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
    def allowed_dispatches(self) -> Tuple:
        """ Returns the allowed dispatches.
        """

        return 'cmg', 'peak_only', 'offpeak-only'

    @property
    def dispatch_mode(self) -> str:
        """ In which dispatch mode you are.
        """

        return self.__dispatch_mode

    @dispatch_mode.setter
    def dispatch_mode(self, new_dispatch: str):
        """ Setting the dispatch, and checking if it's allowed.
        """

        assert self.__dispatch_mode in self.allowed_dispatches, f'Disaptch mode {new_dispatch} not one of allowed dispatches: {self.allowed_dispatches}'

        self.__dispatch_mode = new_dispatch

    def _start_shut_dispatch(self, cs):
        """ Dispatch decision according to shut hours, total starts.

        :param cs: current state vector of the power plant.
        """

        tp = self.tolling_params

        if not self.cuda_ind:
            cs['can_start'] = (cs['total_starts'] < tp['maxMonthlyStarts']) & \
                              (cs['hours_shut'] >= tp['minDownTime'])
        else:  # this kernel implements exactly what is above 3 lines
            cs['can_start'] = cuda_ops.comp_two_arrays_and(cs['total_starts']
                                                           , cs['hours_shut']
                                                           , tp['maxMonthlyStarts']
                                                           , tp['minDownTime'])

    def _peak_only_startup(self, cs):
        """ Startup only at peak times.

        :param cs: current state vector
        """

        tp = self.tolling_params

        cnd_1 = cs['total_starts'] < tp['maxMonthlyStarts']
        cnd_2 = cs['block_name'] == 'peak'
        cnd_3 = cs['hours_shut'] >= tp['minDownTime']

        cs['can_start'] = cnd_1 & cnd_2 & cnd_3 if not self.cuda_ind else cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def _offpeak_only_startup(self, cs : Dict[str, Any]) -> None:
        """ Startup only at peak times, updates the cs, current state variable accordingly.

        :param cs: current state variable.
        """

        tp = self.tolling_params

        cnd_1 = cs['total_starts'] < tp['maxMonthlyStarts']
        cnd_2 = cs['block_name'] != 'peak'
        cnd_3 = cs['hours_shut'] >= tp['minDownTime']

        cs['can_start'] = cnd_1 & cnd_2 & cnd_3 if not self.cuda_ind else cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def _startup_decision(self, cs):
        """ Decision whether to start up.

        :param cs: current state
        """

        dispatch_mode = self.dispatch_mode

        if dispatch_mode == 'cmg':
            self._start_shut_dispatch(cs)

        elif dispatch_mode == 'peak_only':
            self._peak_only_startup(cs)

        elif dispatch_mode == 'offpeak_only':
            self._offpeak_only_startup(cs)

    def _forced_startup(self
                        , curr_state   : Dict[str, Any]
                        , power_prices : np.ndarray
                        , fuel_prices  : np.ndarray
                        , dv            = None):
        """ Try to force power plant to start, updates the curr_state variable accordingly.

        :param cs: current state object
        :param power_prices: vector of power prices
        :param fuel_prices: vector of fuel prices
        :param dv: decision variable
        """

        dispatch_mode = self.dispatch_mode

        if dispatch_mode == 'mrg':
            decision_1 = power_prices - self.tolling_params['hrAtMax'] * fuel_prices
            curr_state['force_start'] = 2 * (decision_1 > dv[1]) + (decision_1 > dv[0]) & (decision_1 < dv[1])
        elif dispatch_mode == 'peak_only':
            curr_state['force_start'].fill(2 * (curr_state['block_name'] == 'peak'))
        elif dispatch_mode == 'offpeak_only':
            curr_state['force_start'].fill(2 * (curr_state['block_name'] != 'peak'))

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

        dispatch_mode = self.dispatch_mode

        assert dispatch_mode in self.allowed_dispatches, f'Dispatch mode {dispatch_mode} not in allowed dispatches: {self.allowed_dispatches}'

        if dispatch_mode == 'cmg':
            self._shut_start_dispatch(cs)

        elif dispatch_mode == 'peak_only':
            self._peak_only_shutdown(cs)

        elif dispatch_mode == 'offpeak_only':
            self._offpeak_only_shutdown(cs)

    def _forced_shutdown(self
                         , cs : Dict
                         , power_prices : Union[np.ndarray, gpa.GPUArray]
                         , fuel_prices  : Union[np.ndarray, gpa.GPUArray]
                         , dv = None):
        """ Decision to forcefully shut down, can take 3 outcomes: 2, 1, 0

        :param cs: current state of the system
        :param power_prices: power price vector
        :param fuel_prices: fuel_price vector
        """

        dispatch_mode = self.dispatch_mode

        if dispatch_mode == 'mrg':
            decision_1 = power_prices - self.tolling_params['hrAtMax'] * fuel_prices
            cs['force_shut'] = 2 * (decision_1 < dv[3]) + (decision_1 > dv[2]) & (decision_1 < dv[3])
        elif dispatch_mode == 'peak_only':
            cs['force_shut'].fill(2 * (cs['block_name'] != 'peak'))
        elif dispatch_mode == 'offpeak_only':
            cs['force_shut'].fill(2 * (cs['block_name'] != 'offpeak'))

    def _set_initial_current_state(self, nb_sims : int) -> Dict[str, Any]:
        """ Set up the initial current state dictionaries.

        :param nb_sims: number of simulations for the month
        :returns: current state dictionary for the beginning of block optimizations.
        """

        dispatch_mode = self.dispatch_mode

        # cs is mnemonic for current state.
        cs = { 'dispatch_mode' :  self.dispatch_mode
             , 'generation'    : 0.
             , 'total_starts'  : 0
             , 'hours_shut'    : 0
             , 'hours_run'     : 0
             , 'df'            : 1.
             , 'state'         : False  # power plant not running
             , 'hours_in_state': 0
             , 'global_starts' : 0
             , }

        if dispatch_mode == 'cmg' or dispatch_mode == 'mrg':
            if not self.cuda_ind:
                cs['force_start'] = np.ones(nb_sims, dtype=np.short)
                cs['force_shut']  = np.ones(nb_sims, dtype=np.short)
                cs['can_start']   = np.empty(nb_sims, dtype=np.short)  # CHECK THIS ONE
                cs['can_shut']    = np.empty(nb_sims, dtype=np.short)
            else:  # cuda
                cs['force_start'] = gpa.empty(nb_sims, dtype=np.int32).fill(1)
                cs['force_shut']  = gpa.empty(nb_sims, dtype=np.int32).fill(1)
                cs['can_start']   = gpa.empty(nb_sims, dtype=bool)
                cs['can_shut']    = gpa.empty(nb_sims, dtype=bool)

        elif dispatch_mode == 'always_run':
            if not self.cuda_ind:  # cpu
                cs['force_start'] = np.empty(nb_sims, dtype=np.short).fill(2)
                cs['force_shut']  = np.zeros(nb_sims, dtype=np.short)
                cs['can_start']   = np.empty(nb_sims, dtype=np.short).fill(1)
                cs['can_shut']    = np.empty(nb_sims, dtype=np.short).fill(0)
            else:  # cuda
                cs['force_shut']  = gpa.empty(dtype=np.int32).fill(0)  # force shut done once
                cs['force_start'] = gpa.empty(dtype=np.int32).fill(2)  # force start set here
                cs['can_shut']    = gpa.zeros(nb_sims, dtype=bool)
                cs['can_start']   = gpa.zeros(nb_sims, dtype=bool) + 1

        else:  # peak & offpeak only
            if not self.cuda_ind:  # cpu
                cs['force_start'] = np.empty(nb_sims, dtype=np.short)
                cs['force_shut']  = np.empty(nb_sims, dtype=np.short)
                cs['can_start']   = np.empty(nb_sims, dtype=np.short)
                cs['can_shut']    = np.empty(nb_sims, dtype=np.short)
            else:  # cuda
                cs['force_start'] = 1  # TODO: FIX FIX FIX FIX
                cs['force_shut']  = 1
                cs['can_start']   = 1
                cs['can_shut']    = 1

        return cs

    @property
    def _opd_f(self):
        """ One period dispatch function selection, depending on whether we are in cuda mode, or not.
        """

        return opd_1fuel.opd_1fuel if not self.cuda_ind else opd_1fuel_cu.opd_kernel

    def __dispatch_update( self
                         , curr_state   : Dict[str, Any]
                         , cash_flows   : np.array
                         , block_hours  : int
                         , block_name   : str
                         , fuel_prices  : np.array
                         , power_prices : np.array
                         , ):
        """ Updates the current state variable.

        :param curr_state: current state variable, gets updated in this routine.
        :param cash_flows: cash flows (one for each simulation)
        :param block_hours: number of hours in this block, e.g. 8
        :param block_name: name of the block (e.g. 'PJMW-PEAK')
        :param fuel_prices: vector of fuel prices
        :param power_prices: vector of power prices.
        """

        curr_state.update( { 'hours_block': block_hours
                           , 'block_name' : block_name})

        tp = self.tolling_params  # abbreviation

        if self.dispatch_mode == 'cmg':
            start_cost_init = tp['fixedStartupCostCold'] + \
                              tp['startFuel'] * (fuel_prices + tp['addFuelCost']) - \
                              tp['hrAtMax'] * power_prices  # TODO: hrAtMax is NOT CORRECT HERE
            # startup shadow prices
            startup_sp_in = start_cost_init / (tp['maxCap'] * tp['startupHorizon'])
        else:
            startup_sp_in = 0.

        # the reason why first are overwritten is that they change very often
        self._startup_decision(curr_state)
        self._shutdown_decision(curr_state)

        # the following updates cs.force_shut, cs.force_start
        self._forced_startup(curr_state, power_prices, fuel_prices)
        self._forced_shutdown(curr_state, power_prices, fuel_prices)

        # dispatch for a block.
        self._block_dispatch( power_prices
                            , fuel_prices
                            , block_hours
                            , startup_sp_in
                            , curr_state
                            , len(power_prices)  # number of simulations
                            , cash_flows )

        return cash_flows

    def dispatch_all( self
                    , tolling_start  : datetime.date
                    , tolling_end    : datetime.date
                    , set_seed       : Optional[int] = None
                    , nb_simulations : int           = 1000 ):
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
        np_gpa = gpa if self.cuda_ind else np

        cash_flows     = np_gpa.empty(nb_simulations)  # cash flow per path
        cash_flows_cum = np_gpa.zeros(nb_simulations)  # cumulative cash flows
        curr_state = self._set_initial_current_state(nb_simulations)

        for date_month in dates_months:  # date_month - beginning of that month
            fuel_process_month  = fuel_process[date_month]  # (list of (block_name, block_hours, block_sims)
            power_process_month = power_processes[date_month]  # same here

            value_per_month = []
            for power_block, fuel_block in zip(power_process_month, fuel_process_month):
                power_block_name, block_hours, power_block_values = power_block
                fuel_block_name , _          , fuel_block_values  = fuel_block

                cash_flows = self.__dispatch_update( curr_state
                                                   , cash_flows
                                                   , block_hours
                                                   , power_block_name
                                                   , fuel_block_values
                                                   , power_block_values
                                                   , )
                cash_flows_cum += cash_flows

                value_per_month.append( (block_hours, (power_block_name, fuel_block_name), cash_flows) )

            dispatch_per_month[date_month] = value_per_month

        return dispatch_per_month

    def _block_dispatch(self
                        , power_prices         : Union[np.ndarray, gpa.GPUArray]
                        , fuel_prices          : Union[np.ndarray, gpa.GPUArray]
                        , block_hours          : int
                        , startup_shadow_price : float
                        , curr_state           : Dict[str, Any]
                        , nb_sims              : int
                        , cash_flows           : np.array ):
        """ Dispatch in a single block, changes the current state as appropriate.

        :param power_prices: vector of power prices
        :param fuel_prices: vector of fuel prices.
        :param block_hours: number of hours in the current block
        :param startup_shadow_price: vector of startup shadow prices.
        :param curr_state: current state, a dictionary of various elements
        :param nb_sims: number of simulations
        :param cash_flows: cash flows to be updated in the block dispatch.
        """

        self._opd_f( power_prices
                   , fuel_prices
                   , self.tolling_params
                   , startup_shadow_price
                   , curr_state
                   , { 'can_start'  : curr_state['can_start']
                     , 'can_shut'   : curr_state['can_shut']
                     , 'force_start': curr_state['force_start']
                     , 'force_shut' : curr_state['force_shut']
                     , }
                   , block_hours
                   , nb_sims
                   , cash_flows )
