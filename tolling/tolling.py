# Tolling model
import mrds.config

import datetime
import numpy as np

from typing import List, Tuple, Union, Dict, Callable, Optional, Any

from mrds.tolling.opd              import opd_1fuel, opd_1fuel_cu
from mrds.tolling.opd.opd_1fuel    import TollingState
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
                        , dcf            = dcf
                        , cuda_ind       = cuda_ind )

        self.fuel_idx           = fuel_idx
        self.tolling_params     = tolling_params

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
    def allowed_dispatches(self) -> Tuple[str, str, str]:
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

    def _start_shut_dispatch(self
                            , total_run_hours : Union[int, np.ndarray]
                            , hours_shut      : int ) -> Union[bool, np.ndarray]:
        """ Dispatch decision according to shut hours, total starts.

        :param total_run_hours: number of hours ran so far
        :param hours_shut: hours that the plant has been shut
        :returns: decision if the power plant can start, should be updated in cs['can_start']
        """

        tp = self.tolling_params

        if not self.cuda_ind:
            return (total_run_hours < tp['maxMonthlyStarts']) & (hours_shut >= tp['minDownTime'])

        # CUDA part: this kernel implements exactly what is above 3 lines
        return cuda_ops.comp_two_arrays_and( total_run_hours
                                           , hours_shut
                                           , tp['maxMonthlyStarts']
                                           , tp['minDownTime'] )

    def _peak_only_startup( self
                          , total_starts : Union[int, np.ndarray]
                          , block_name   : str
                          , hours_shut   : Union[int, np.ndarray] ) -> Union[bool, np.ndarray]:
        """ Startup only at peak times.

        :param total_starts: total number of starts
        :param block_name: current block name
        :parma hours_shut: number of hours shut
        :returns: whether the power plant can start in this block
        """

        tp = self.tolling_params

        cnd_1 = total_starts < tp['maxMonthlyStarts']
        cnd_2 = block_name == 'peak'
        cnd_3 = hours_shut >= tp['minDownTime']

        if not self.cuda_ind:
            return cnd_1 & cnd_2 & cnd_3

        # cuda section
        return gpa.minimum(gpa.minimum(cnd_1, cnd_2), cnd_3)
        # TODO: better version of the above.
        # cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def _offpeak_only_startup( self
                             , total_starts : Union[int, np.ndarray]
                             , block_name : str
                             , hours_shut : Union[int, np.ndarray] ) -> np.ndarray:
        """ Startup only at peak times, updates the cs, current state variable accordingly.

        :param total_starts: total number of starts per month so far.
        :param block_name: current block name
        :param hours_shut: number of hours shut.
        :returns: array indicating whether the power plant can start.
        """

        tp = self.tolling_params

        cnd_1 = total_starts < tp['maxMonthlyStarts']
        cnd_2 = block_name != 'peak'
        cnd_3 = hours_shut >= tp['minDownTime']

        if not self.cuda_ind:
            return cnd_1 & cnd_2 & cnd_3

        # Cuda version
        return gpa.minimum(gpa.minimum(cnd_1, cnd_2), cnd_3)
        # return cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def _startup_decision(self, cs):
        """ Decision whether to start up, depending on the dispatch method.

        :param cs: current state
        """

        dispatch_mode = self.dispatch_mode

        assert dispatch_mode in self.allowed_dispatches, f'Dispatch mode {dispatch_mode} not allowed. Choose among {self.allowed_dispatches}'

        if dispatch_mode == 'cmg':
            return self._start_shut_dispatch(cs['hours_run'], cs['hours_shut'])

        if dispatch_mode == 'peak_only':
            return self._peak_only_startup(cs['total_starts'], cs['block_name'], cs['hours_shut'])

        if dispatch_mode == 'offpeak_only':
            return self._offpeak_only_startup(cs['total_starts'], cs['block_name'], cs['hours_shut'])

    def _forced_startup(self
                        , block_name   : str
                        , power_prices : np.ndarray
                        , fuel_prices  : np.ndarray
                        , dv            = None):
        """ Try to force power plant to start, updates the curr_state variable accordingly.

        :param block_name: current block name
        :param power_prices: vector of power prices
        :param fuel_prices: vector of fuel prices
        :param dv: decision variable
        """

        dispatch_mode = self.dispatch_mode

        assert dispatch_mode in self.allowed_dispatches, f'Dispatch mode {dispatch_mode} must be one of {self.allowed_dispatches}'

        if dispatch_mode == 'cmg':
            return False

        if dispatch_mode == 'mrg':
            decision_1 = power_prices - self.tolling_params['hrAtMax'] * fuel_prices
            # TODO: What is dv[0], dv[1] etc.
            return 2 * (decision_1 > dv[1]) + (decision_1 > dv[0]) & (decision_1 < dv[1])

        if dispatch_mode == 'peak_only':
            res = np.empty(len(power_prices))
            res.fill(2 * (block_name == 'peak'))

            return res

        elif dispatch_mode == 'offpeak_only':
            res = np.empty(len(power_prices))
            res.fill(2 * (block_name != 'peak'))
            return res

    def _shut_start_dispatch(self, hours_run : Union[int, np.ndarray]) -> Union[bool, np.ndarray]:
        """ Indicator function if the power plant can shut down. The condition is if hours_run is bigger
            than the minimum runtime. Replaces the 'can_shut' part of the current state dictionary with the new state.

        :param hours_run: number of hours that the power plant has ran so far.
        :returns: True if it can shut down or not.
        """

        tp = self.tolling_params

        if not self.cuda_ind:
            return hours_run >= tp['minRunTime']

        return cuda_ops.comp_array_number(hours_run, tp['minRunTime'], op='larger', dtype='int32')

    def _peak_only_shutdown(self, block_name : str, hours_run : Union[int, np.ndarray]) -> Union[bool, np.ndarray]:
        """ Whether the power plant can shut at a particular block.

        """

        cnd_2 = block_name != 'peak'
        cnd_3 = hours_run >= self.tolling_params['minRunTime']

        if not self.cuda_ind:
            return cnd_2 & cnd_3

        # TODO: THIS BELOW IS WRONG - FIX FIX FIX
        return cuda_ops.min_int_two(cnd_2, cnd_3.astype(np.int32), cs['can_shut'])

    def _offpeak_only_shutdown(self, block_name : str, hours_run : Union[int, np.ndarray]) -> None:

        cnd_2 = block_name == 'peak'
        cnd_3 = hours_run  >= self.tolling_params['minRunTime']

        if not self.cuda_ind:
            return cnd_2 & cnd_3  # cs['can_shut']

        # cuda version
        return cuda_ops.min_int_two(cnd_2, cnd_3, cs['can_shut'])  # TODO: that's wrong - FIX THIS HERE

    def _shutdown_decision(self, block_name : str, hours_run : int) -> Union[bool, np.ndarray]:
        """ Decision whether it is sensible to shut down

        :param block_name: block name
        :param hours_run: hours run so far.
        """

        dispatch_mode = self.dispatch_mode

        assert dispatch_mode in self.allowed_dispatches, f'Dispatch mode {dispatch_mode} not in allowed dispatches: {self.allowed_dispatches}'

        if dispatch_mode == 'cmg':
            return self._shut_start_dispatch(hours_run)

        if dispatch_mode == 'peak_only':
            return self._peak_only_shutdown(block_name, hours_run)

        if dispatch_mode == 'offpeak_only':
            return self._offpeak_only_shutdown(block_name, hours_run)

    def _forced_shutdown(self
                         , block_name   : str
                         , power_prices : Union[np.ndarray, gpa.GPUArray]
                         , fuel_prices  : Union[np.ndarray, gpa.GPUArray]
                         , dv = None) -> Union[bool, np.ndarray]:
        """ Decision to forcefully shut down, can take 3 outcomes: 2, 1, 0

        :param block_name: current block name
        :param power_prices: power price vector
        :param fuel_prices: fuel_price vector
        :param dv: decision variable TODO: EXPLAIN BETTER
        :returns: indicator whether the power plant should force shutdown.
        """

        dispatch_mode = self.dispatch_mode

        assert dispatch_mode in self.allowed_dispatches, f'Dispatch mode {dispatch_mode} not among the allowed ones: {self.allowed_dispatches}'

        if dispatch_mode == 'cmg':  # TODO: FIX THIS LATER, FOR NOW JUST RETURN False
            return False

        if dispatch_mode == 'mrg':
            decision_1 = power_prices - self.tolling_params['hrAtMax'] * fuel_prices
            return 2 * (decision_1 < dv[3]) + (decision_1 > dv[2]) & (decision_1 < dv[3])

        if dispatch_mode == 'peak_only':
            return self.__const_array(len(power_prices), 2 if block_name != 'peak' else 0, np.short)

        if dispatch_mode == 'offpeak_only':
            return self.__const_array(len(power_prices), 2 if block_name != 'offpeak' else 0, np.short)

    def __const_array(self, size : int, value : float, dtype_=bool) -> Union[np.ndarray, gpa.GPUArray]:
        """ Returns a bool array of size size, with all values set to value.

        :param size: size of the array
        :param value: value the array is set to
        :param dtype_: type of the array to be generated
        :returns: array of size size and value set to value, either np.array or gpu array
        """

        res = np.empty(size, dtype=dtype_) if not self.cuda_ind else gpa.empty(size, dtype=dtype_)
        res.fill(value)

        return res

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
             , 'hours_shut'    : 1000  # large number
             , 'hours_run'     : 0
             , 'df'            : 1.
             , 'state'         : TollingState.NOT_RUNNING
             , 'hours_in_state': 0
             , 'global_starts' : 0
             , 'startup_sp'    : 0.  # TODO: CHECK THIS - INITIAL SHADOW PRICE
             , }

        if dispatch_mode == 'cmg' or dispatch_mode == 'mrg':

            if not self.cuda_ind:
                cs['force_start'] = self.__const_array(nb_sims, False)
                cs['force_shut']  = self.__const_array(nb_sims, False)
                cs['can_start']   = self.__const_array(nb_sims, True)
                cs['can_shut']    = self.__const_array(nb_sims, True)

            # if not self.cuda_ind:
            #     cs['force_start'] = np.ones(nb_sims, dtype=np.short)
            #     cs['force_shut']  = np.ones(nb_sims, dtype=np.short)
            #     cs['can_start']   = np.empty(nb_sims, dtype=np.short)  # CHECK THIS ONE
            #     cs['can_shut']    = np.empty(nb_sims, dtype=np.short)
            #
            else:  # cuda
                cs['force_start'] = self.__const_array(nb_sims, 1, dtype_=np.int32)
                cs['force_shut']  = self.__const_array(nb_sims, 1, dtype_=np.int32)
                cs['can_start']   = gpa.empty(nb_sims, dtype=bool)
                cs['can_shut']    = gpa.empty(nb_sims, dtype=bool)

        elif dispatch_mode == 'always_run':
            if not self.cuda_ind:  # cpu
                cs['force_start'] = self.__const_array(nb_sims, 2, dtype_=np.short)
                cs['force_shut']  = np.zeros(nb_sims, dtype=np.short)
                cs['can_start']   = np.ones(nb_sims, dtype=np.short)
                cs['can_shut']    = np.zeros(nb_sims, dtype=np.short)
            else:  # cuda
                cs['force_shut']  = gpa.zeros(1, dtype=np.int32)  # force shut done once
                cs['force_start'] = gpa.zeros(1, dtype=np.int32).fill(2)  # force start set here TODO: THIS IS ALL WRONG
                cs['can_shut']    = gpa.zeros(nb_sims, dtype=bool)
                cs['can_start']   = self.__const_array(nb_sims, True, dtype=bool)

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

        :returns: function executing one-block dispatch.
        """

        return opd_1fuel.opd_1fuel if not self.cuda_ind else opd_1fuel_cu.one_period_dispatch

    def __compute_shadow_cost(self, fuel_prices : np.ndarray, power_prices : np.ndarray) -> Union[np.ndarray, float]:
        """ Computes the shadow cost for the model.

        :param fuel_prices: vector of fuel prices.
        :param power_prices: vector of power prices.
        :returns: shadow cost of the model.
        """

        tp = self.tolling_params

        if self.dispatch_mode == 'cmg':
            start_cost_init = tp['fixedStartupCostCold'] + \
                              tp['startFuel'] * (fuel_prices + tp['addFuelCost']) - \
                              tp['hrAtMax'] * power_prices  # TODO: hrAtMax is NOT CORRECT HERE
            # startup shadow prices
            return start_cost_init / (tp['maxCap'] * tp['startupHorizon'])

        return 0.

    def __dispatch_update( self
                         , curr_state   : Dict[str, Any]
                         , block_hours  : int
                         , block_name   : str
                         , fuel_prices  : np.array
                         , power_prices : np.array
                         , ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Updates the current state variable.

        :param curr_state: current state variable, gets updated in this routine.
        :param block_hours: number of hours in this block, e.g. 8
        :param block_name: name of the block (e.g. 'PJMW-PEAK')
        :param fuel_prices: vector of fuel prices
        :param power_prices: vector of power prices.
        :returns:
        """

        curr_state.update( { 'hours_block': block_hours
                           , 'block_name' : block_name})

        curr_state['startup_sp']  = self.__compute_shadow_cost(fuel_prices, power_prices)
        curr_state['can_start']   = self._startup_decision(curr_state)
        curr_state['can_shut' ]   = self._shutdown_decision(block_name, curr_state['hours_run'])
        curr_state['force_start'] = self._forced_startup(block_name, power_prices, fuel_prices)
        curr_state['force_shut']  = self._forced_shutdown(block_name, power_prices, fuel_prices)

        # dispatch for a block, updates both curr_state, and cash_flows
        cash_flows, new_curr_state = self._block_dispatch( power_prices
                                                         , fuel_prices
                                                         , block_hours
                                                         , curr_state
                                                         , len(power_prices))   # number of simulations

        return cash_flows, new_curr_state

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
                                                , set_seed        = set_seed
                                                , hours_partition = fuel_hours_partition )

        power_processes = self.simulate_spot_blocks( [fwd_curve.fwd_name for fwd_curve in self.fwd_curves]
                                                   , nb_simulations
                                                   , tolling_start
                                                   , tolling_end
                                                   , set_seed = set_seed )

        # dates_months = list(fuel_process.keys())  # same as power_process.keys()

        dispatch_per_month = {}
        np_gpa     = gpa if self.cuda_ind else np
        curr_state = self._set_initial_current_state(nb_simulations)

        # for date_month in dates_months:  # date_month - beginning of that month
        for (date_month_fuel, fuel_process_month), (date_month_power, power_process_month) in zip(fuel_process, power_processes):  # date_month - beginning of that month
            assert date_month_fuel == date_month_power, f'Start dates for power and fuel month are different: {date_month_power, date_month_fuel}'
            # fuel_process_month  = fuel_process[date_month]  # (list of (block_name, block_hours, block_sims)
            # power_process_month = power_processes[date_month]  # same here

            cash_flows_cum = np_gpa.zeros(nb_simulations)  # cumulative cash flows

            value_per_month = []
            for power_block, fuel_block in zip(power_process_month, fuel_process_month):
                power_block_name, block_hours, power_block_values = power_block
                fuel_block_name , _          , fuel_block_values  = fuel_block

                # cashflows and new current state, which updates the old curr_state
                cash_flows, curr_state = self.__dispatch_update( curr_state
                                                               , block_hours
                                                               , power_block_name
                                                               , fuel_block_values
                                                               , power_block_values
                                                               , )
                cash_flows_cum += cash_flows
                # TODO: REMOVE per-block cash flows when this works.
                value_per_month.append( (block_hours, (power_block_name, fuel_block_name), cash_flows) )

            dispatch_per_month[date_month_fuel] = (cash_flows_cum, value_per_month)

        return dispatch_per_month

    def _block_dispatch(self
                        , power_prices         : Union[np.ndarray, gpa.GPUArray]
                        , fuel_prices          : Union[np.ndarray, gpa.GPUArray]
                        , block_hours          : int
                        , curr_state           : Dict[str, Any]
                        , nb_sims              : int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Dispatch in a single block, changes the current state as appropriate.

        :param power_prices: vector of power prices
        :param fuel_prices: vector of fuel prices.
        :param block_hours: number of hours in the current block
        :param curr_state: current state, a dictionary of various elements
        :param nb_sims: number of simulations
        :returns: updated cash-flows, and updated current state.
        """

        return self._opd_f( power_prices
                          , fuel_prices
                          , self.tolling_params
                          , curr_state
                          , { 'can_start'  : curr_state['can_start']
                            , 'can_shut'   : curr_state['can_shut']
                            , 'force_start': curr_state['force_start']
                            , 'force_shut' : curr_state['force_shut']
                            , }
                          , block_hours
                          , nb_sims )
