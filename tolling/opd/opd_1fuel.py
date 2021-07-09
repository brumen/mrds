#
#  One period dispatch accounting.
#

import numpy as np

from logging import getLogger
from typing  import Any, Dict, Union

logger = getLogger(__name__)

try:
    import pycuda.autoinit
except:
    logger.info('CUDA functionality not enabled.')
import pycuda.gpuarray as gpa

from mrds.tolling.opd   import opd_avx
from pycuda.gpuarray    import GPUArray, where as gpawhere

SMALL_EPS = 1e-5

from mrds.tolling.tolling_states import TollingState, const_array


def compare_with_numpy(toll_fct):

    def wrapped_toll_fct(*args, **kwargs):
        gpa_res = toll_fct(*args, cuda_ind=True)
        np_args = [arg if not isinstance(arg, GPUArray) else arg.get()
                   for arg in args]
        np_res  = toll_fct(*np_args, cuda_ind=False)
        gpa_np_res = gpa_res.get()  # this is a np-array

        comparison = all([gpa_elt == np_elt for gpa_elt, np_elt in zip(gpa_np_res, np_res)])
        if not comparison:
            print('WRONG')

        # assert comparison, f'Problems in function {toll_fct.__name__}'

        return gpa_res

    return wrapped_toll_fct


# @compare_with_numpy
def starts_indicator( curr_state : Union[np.array, GPUArray]
                    , new_state  : Union[np.array, GPUArray]
                    , cuda_ind   : bool = False  ) -> Union[np.array, GPUArray]:
    """ Indicator function of when the power plant starts.

    :param curr_state: current state array of TollingStates
    :param new_state: new state array of TollingStates
    :param cuda_ind: indicator for cuda, default = False
    :returns: array of booleans of whether the start occurred or not.
    """

    if not cuda_ind:
        return (curr_state == TollingState.NOT_RUNNING) & (
                (new_state == TollingState.MAX_DISPATCH) | (new_state == TollingState.MIN_DISPATCH))

    # cuda stuff
    return (curr_state == TollingState.NOT_RUNNING.value) & (
                (new_state == TollingState.MAX_DISPATCH.value) | (new_state == TollingState.MIN_DISPATCH.value))


def shuts_indicator( curr_state : Union[np.ndarray, GPUArray]
                   , new_state  : Union[np.array, GPUArray]
                   , cuda_ind   : bool = False ) -> np.array:
    """ Indicator if the power plant was shutdown.

    :param curr_state: array of current states.
    :param new_state: array of new states.
    :param cuda_ind: indicator of the cuda.
    :returns: array of states if the power plant
    """

    if not cuda_ind:
        return ((curr_state == TollingState.MAX_DISPATCH) | (curr_state == TollingState.MIN_DISPATCH)) & (
                    new_state == TollingState.NOT_RUNNING)

    # cuda
    return ((curr_state == TollingState.MAX_DISPATCH.value) | (curr_state == TollingState.MIN_DISPATCH.value)) & (new_state == TollingState.NOT_RUNNING.value)


def still_running_indicator( curr_state : Union[np.array, GPUArray]
                           , new_state  : Union[np.array, GPUArray]
                           , cuda_ind   : bool = False  ) -> Union[np.array, GPUArray]:
    """ Indicator if the power plant was shutdown.

    :param curr_state: array of current states.
    :param new_state: array of new states.
    :param cuda_ind
    :returns: array of states if the power plant
    """

    if not cuda_ind:
        return ( (curr_state == TollingState.MIN_DISPATCH) | (curr_state == TollingState.MAX_DISPATCH)) & \
               ( (new_state == TollingState.MIN_DISPATCH) | (new_state == TollingState.MAX_DISPATCH))

    # cuda ind
    return ( (curr_state == TollingState.MIN_DISPATCH.value) | (curr_state == TollingState.MAX_DISPATCH.value)) & \
               ( (new_state == TollingState.MIN_DISPATCH.value) | (new_state == TollingState.MAX_DISPATCH.value))


def do_startup_f( can_start             : Union[bool, np.ndarray, GPUArray]
                , force_start           : Union[bool, np.ndarray, GPUArray]
                , is_startup_profitable : Union[bool, np.ndarray, GPUArray]
                , cuda_ind              : bool =False ) -> Union[bool, np.ndarray, GPUArray]:
    """ Decision whether to do the startup or not.

    :param can_start: indicator whether the power plant can start.
    :param force_start: whether it is forced to start
    :param is_startup_profitable: indicator whether the startup is profitable at all.
    :param cuda_ind: indicator whether you want cuda or not.
    :returns: indicator whether you should do the startup or not.
    """

    if not cuda_ind:
        return can_start & ( force_start | is_startup_profitable )

    # cuda section
    if isinstance(can_start, bool):
        # TODO: THIS IS WRONG HERE - FIX THIS
        return (force_start or is_startup_profitable) if can_start else False

    # GPUArray is can_start
    return gpawhere( can_start
                   , force_start or is_startup_profitable
                   , False )


def gpa_where( cond       : Union[np.ndarray, GPUArray]
             , cond_true  : Union[float, int, np.short, np.ndarray, GPUArray]
             , cond_false : Union[float, int, np.short, np.ndarray, GPUArray] ):
    """ Takes cond_true value if cond is true, and cond_false value if cond is false.

    :param cond: array of boolean values.
    :param cond_true: true value
    :param cond_false: false value.
    :returns: an array taking respective values from cond_true, cond_false
    """

    if isinstance(cond, bool):
        return cond_true if cond else cond_false

    return gpawhere(cond, cond_true, cond_false)


def new_state_f( curr_state  : Union[bool, np.ndarray, GPUArray]
               , do_startup  : Union[bool, np.ndarray, GPUArray]
               , do_shutdown : Union[bool, np.ndarray, GPUArray]
               , cuda_ind    : bool = False ) -> Union[bool, np.ndarray, GPUArray]:
    """ Computes the new state of the system, from the old (current state), and decision variables.

    :param curr_state: vector of current state.
    :param do_startup: whether to do the startup
    :param do_shutdown: whether to shut it down.
    :param cuda_ind: indicator for cuda.
    :returns: new state of the system.
    """

    if not cuda_ind:
        return  np.where( (curr_state == TollingState.NOT_RUNNING) & do_startup
                        , TollingState.MAX_DISPATCH
                        , np.where( (curr_state == TollingState.MAX_DISPATCH) & do_shutdown
                                  , TollingState.NOT_RUNNING
                                  , curr_state )  # remain in the same state
                        )

    # cuda version
    max_dispatch = const_array(len(curr_state), TollingState.MAX_DISPATCH.value, np.short, cuda_ind=True)
    not_running  = const_array(len(curr_state), TollingState.NOT_RUNNING.value , np.short, cuda_ind=True)

    return gpa_where( (curr_state == TollingState.NOT_RUNNING.value) and do_startup
                    , max_dispatch
                    , gpa_where( (curr_state == TollingState.MAX_DISPATCH.value) and do_shutdown
                               , not_running
                               , curr_state )  # remain in state
                    )


def curr_generation_f(new_state : Union[np.ndarray, GPUArray], max_cap : float, min_disp : float, cuda_ind : bool = False):
    """ Computes current generation.

    :param new_state: new state of the system.
    :param max_cap: maximum capacity of the power plant.
    :param min_disp: minimum dispatch
    :param cuda_ind: indicator for cuda
    :returns: current generation of the power plant.
    """

    if not cuda_ind:
        return np.where( new_state == TollingState.NOT_RUNNING
                              , 0.
                              , np.where( new_state == TollingState.MAX_DISPATCH
                                        , max_cap
                                        , min_disp ) )

    # cuda result
    return gpa_where( new_state == TollingState.NOT_RUNNING.value
                    , gpa.zeros(len(new_state), dtype=float)
                    , gpa_where( new_state == TollingState.MAX_DISPATCH.value
                               , const_array(len(new_state), max_cap , dtype_=float, cuda_ind = True )
                               , const_array(len(new_state), min_disp, dtype_=float, cuda_ind = True )
                               , )
                    , )


def ramp_cost_f( curr_state     : Union[np.ndarray, GPUArray]
               , new_state      : Union[np.ndarray, GPUArray]
               , ramp_up_cost   : float
               , ramp_down_cost : float
               , cuda_ind       : bool = False ) -> Union[np.ndarray, GPUArray]:
    """ Cost of ramping up or down.

    :param curr_state: current state of the system
    :param new_state: new state of the system.
    :param ramp_up_cost: ramp up costs.
    :param ramp_down_cost: ramp down costs.
    :param cuda_ind: indicator for cuda
    :returns: costs of ramping up.
    """

    if not cuda_ind:
        return np.where( (curr_state == TollingState.MIN_DISPATCH) & (new_state == TollingState.MAX_DISPATCH)
                       , ramp_up_cost
                       , np.where( (curr_state == TollingState.MAX_DISPATCH) & (new_state == TollingState.MIN_DISPATCH)
                                 , ramp_down_cost
                                 , 0. ) )

    # CUDA section
    return gpa_where( (curr_state == TollingState.MIN_DISPATCH.value) & (new_state == TollingState.MAX_DISPATCH.value)
                    , ramp_up_cost
                    , gpa_where( ( curr_state == TollingState.MAX_DISPATCH.value) & ( new_state == TollingState.MIN_DISPATCH.value)
                               , ramp_down_cost
                               , 0. )
                    , )


def startup_cost_f( starts
                  , is_cold_start    : Union[bool, np.ndarray, GPUArray]
                  , cold_start_costs
                  , hot_start_costs
                  , cuda_ind         : bool = False):

    if not cuda_ind:
        return np.where( starts
                       , np.where( is_cold_start, cold_start_costs, hot_start_costs)
                       , 0. )

    # cuda section
    if not isinstance(is_cold_start, GPUArray):
        return gpa_where( starts
                        , cold_start_costs if is_cold_start else hot_start_costs
                        , 0. )

    # GPU array example
    return gpa_where( starts
                    , gpa_where(is_cold_start, cold_start_costs, hot_start_costs )
                    , gpa.zeros(is_cold_start.size, dtype=cold_start_costs.dtype)
                    , )


def opd_1fuel( power_prices   : np.ndarray
             , fuel_prices    : np.ndarray
             , tolling_params : Dict[str, Any]
             , curr_state     : Dict[str, Any]
             , curr_decision  : Dict[str, np.ndarray]
             , hours_in_block : int
             , nb_paths       : int
             , cuda_ind       : bool = False ):
    """ Computes one block tolling optimization.

    :param power_prices: power prices
    :param fuel_prices: fuel prices
    :param tolling_params: parameters of the tolling
    :param curr_state: current state vector
    :param curr_decision: current decision vector
    :param hours_in_block: hours for current block
    :param nb_paths: number of paths for , also the length of the power & fuel price vectors.
    :param cuda_ind: indicator whether to use cuda.
    """

    hr_at_max       = tolling_params['hrAtMax']
    hr_at_min       = tolling_params['hrAtMin']
    max_cap         = tolling_params['maxCap']
    min_disp        = tolling_params['minDisp']
    start_fuel      = tolling_params['startFuel']
    start_fuel_cold = tolling_params['startFuelCold']
    add_fuel_cost   = tolling_params['addFuelCost']
    VC              = tolling_params['VC']
    # ramp_rate       = tolling_params['rampRate']
    shutdown_sp_in  = tolling_params['shutdownSPin']  # TODO: THIS IS WRONG, THIS HAS TO BE UPDATED
    # min_downtime    = tolling_params['minDownTime']
    # min_runtime     = tolling_params['minRunTime']
    fixed_startup_cost = tolling_params['fixedStartupCost']
    fixed_startup_cost_cold = tolling_params['fixedStartupCostCold']
    # max_monthly_starts = tolling_params['maxMonthlyStarts']
    cold_startup = tolling_params['coldStartup']
    startup_horizon = tolling_params['startupHorizon']
    shutdown_horizon = tolling_params['shutdownHorizon']
    ramp_up_sp_in    = tolling_params['rampUpSPin']
    ramp_down_sp_in  = tolling_params['rampDownSPin']
    ramp_up_cost     = tolling_params['rampUpCost']
    ramp_down_cost   = tolling_params['rampDownCost']
    ramp_up_horizon  = tolling_params['rampUpHorizon']
    ramp_down_horizon = tolling_params['rampDownHorizon']

    where = np.where if not cuda_ind else gpa_where

    # what kind of state the power plant is in, type = TollingState
    state_state = curr_state['state']  # np.array[dtype=TollingState]

    # marginal cost at max
    optimal_marginal_cost_at_max = (fuel_prices + add_fuel_cost) * hr_at_max + VC
    optimal_marginal_cost_at_min = (fuel_prices + add_fuel_cost) * hr_at_min + VC

    # ramping costs
    generation_smaller_maxcap = curr_state['generation'] < max_cap
    ramp_up_to_max_cost = generation_smaller_maxcap * (ramp_up_sp_in + ramp_up_cost / (max_cap * ramp_up_horizon))
    generation_larger_mindisp = curr_state['generation'] > min_disp
    ramp_down_to_min_cost = generation_larger_mindisp * (ramp_down_sp_in + ramp_down_cost / (min_disp * ramp_down_horizon))
    # run_at_min_index = whether it's better to run at minimum dispatch, than at maximum dispatch np.array[bool]
    run_at_min_index = max_cap * (power_prices - optimal_marginal_cost_at_max - ramp_up_to_max_cost) < \
                       min_disp * (power_prices - optimal_marginal_cost_at_min - ramp_down_to_min_cost)

    # compute total startup costs
    is_cold_start     = curr_state['hours_shut'] >= cold_startup

    fixed_and_fuel_startup_cost = where( is_cold_start
                                       , fixed_startup_cost_cold + start_fuel_cold * fuel_prices
                                       , fixed_startup_cost + start_fuel * fuel_prices
                                       , )

    # startup shadow price
    startup_sp = fixed_and_fuel_startup_cost / (startup_horizon * max_cap)  # Startup shadow price TODO: CHECK IF THIS MAKES SENSE
    is_startup_profitable = power_prices - optimal_marginal_cost_at_max - startup_sp
    is_startup_profitable = is_startup_profitable > 0.

    do_startup = do_startup_f(curr_decision['can_start'], curr_decision['force_start'], is_startup_profitable, cuda_ind=cuda_ind)

    actual_gen_profit = where( run_at_min_index
                             , (power_prices - optimal_marginal_cost_at_min) * min_disp
                             , (power_prices - optimal_marginal_cost_at_max) * max_cap
                             , )

    shutdown_gen_profit = shutdown_horizon * actual_gen_profit
    shut_cost_sp = shutdown_horizon * shutdown_sp_in * max_cap
    is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp)  # TODO: THIS IS WRONG

    # do_shutdown = curr_decision['can_shut'] & ( curr_decision['force_shut'] | is_shutdown_profitable )
    do_shutdown = do_startup_f(curr_decision['can_shut'], curr_decision['force_shut'], is_shutdown_profitable, cuda_ind=cuda_ind)

    # compute new state
    new_state = new_state_f(state_state, do_startup, do_shutdown, cuda_ind=cuda_ind)

    # Generation accounting
    curr_generation = curr_generation_f(new_state, max_cap, min_disp, cuda_ind=cuda_ind)

    # generation_change = curr_generation - curr_state['generation']
    # ramping_adjustment = (0.5 / (ramp_rate * hours_in_block)) * np.abs(generation_change) * generation_change
    # ramping_adjustment = 0.  # TODO: THIS IS WRONG CHECK HERE
    # curr_generation -= ramping_adjustment
    curr_energy = curr_generation * hours_in_block
    revenue = curr_energy * power_prices
    fuel_cost = curr_energy * (fuel_prices + add_fuel_cost) * where(run_at_min_index, hr_at_min, hr_at_max)

    # new starts and new shutdowns.
    starts = starts_indicator(state_state, new_state, cuda_ind=cuda_ind)
    startup_cost = startup_cost_f( starts
                                 , is_cold_start
                                 , fixed_startup_cost_cold + fuel_prices * start_fuel_cold
                                 , fixed_startup_cost + fuel_prices * start_fuel
                                 , cuda_ind = cuda_ind )

    ramp_cost = ramp_cost_f(state_state, new_state, ramp_up_cost, ramp_down_cost, cuda_ind = cuda_ind)

    # cashflow = revenue - (totalCost = fuel_cost + (variable_cost = VC * curr_energy) + startup_cost + ramp_cost)
    if not cuda_ind:
        cashflow = np.empty(len(revenue), dtype=float)
        opd_avx.add4(revenue, fuel_cost, VC * curr_energy, startup_cost, ramp_cost, cashflow, nb_paths)
    else:
        cashflow = revenue - fuel_cost - VC * curr_energy - startup_cost - ramp_cost

    # new state parameters
    curr_state_update = { 'hours_in_state': where( new_state == state_state  # no state change
                                                 , curr_state['hours_in_state'] + hours_in_block
                                                 , hours_in_block )
                        , 'generation'    : curr_generation
                        , 'total_starts'  : curr_state['total_starts'] + starts
                        , 'hours_shut'    : where( ((new_state == TollingState.NOT_RUNNING) & (state_state == TollingState.NOT_RUNNING)) if not cuda_ind else
                                                       ((new_state == TollingState.NOT_RUNNING.value) & (state_state == TollingState.NOT_RUNNING.value))
                                                     , curr_state['hours_shut'] + hours_in_block
                                                     , const_array(len(new_state), 0, dtype_=np.int, cuda_ind=cuda_ind)
                                                     , )
                        , 'hours_run'     : where( still_running_indicator(state_state, new_state, cuda_ind = cuda_ind)
                                                 , curr_state['hours_run'] + hours_in_block
                                                 , const_array(len(new_state), 0, dtype_=np.int, cuda_ind=cuda_ind)
                                                 , )
                        , 'global_starts' : curr_state['global_starts'] + starts
                        , 'state'         : new_state
                        , }

    return cashflow, curr_state_update
