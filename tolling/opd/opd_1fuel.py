#
#  One period dispatch accounting.
#

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpa

from typing import Any, Dict, Union
from enum   import Enum

from mrds.tolling.opd   import opd_avx
from cuda.cuda.cuda_ops import ( selection_kernel
                               , negate_bool
                               , bigger_gpa
                               , equal_gpa
                               , selection_kernel_singles
                               , selection_kernel_first_single
                               , equal_bools
                               , )

from pycuda.gpuarray import  GPUArray

SMALL_EPS = 1e-5


class TollingState(Enum):
    """ State that the power plant can be in.
    """

    NOT_RUNNING  = 1
    MIN_DISPATCH = 2
    MAX_DISPATCH = 3


def invert_bool(x : Union[np.ndarray, bool, GPUArray]) -> Union[np.ndarray, bool, GPUArray]:
    """ Invert a bool vector or variable.

    :param x: initial bool array
    :returns: array of inverted booleans.
    """

    if isinstance(x, np.ndarray):
        return ~x

    if isinstance(x, GPUArray):
        return negate_bool(x)

    # here x is of bool type
    return not x


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
    return equal_gpa(curr_state, TollingState.NOT_RUNNING.value) and (
                (equal_gpa(new_state, TollingState.MAX_DISPATCH.value) or equal_gpa(new_state, TollingState.MIN_DISPATCH.value)) )


def shuts_indicator( curr_state : Union[np.ndarray, GPUArray]
                   , new_state  : Union[np.array, GPUArray]
                   , cuda_ind   : bool = False ) -> np.array:
    """ Indicator if the power plant was shutdown.

    :param curr_state: array of current states.
    :param new_state: array of new states.
    :returns: array of states if the power plant
    """

    if not cuda_ind:
        return ((curr_state == TollingState.MAX_DISPATCH) | (curr_state == TollingState.MIN_DISPATCH)) & (new_state == TollingState.NOT_RUNNING)

    # cuda
    return (equal_gpa(curr_state, TollingState.MAX_DISPATCH.value) or equal_gpa(curr_state, TollingState.MIN_DISPATCH.value)) and \
           equal_gpa(new_state, TollingState.NOT_RUNNING.value)


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
    return ( equal_gpa(curr_state, TollingState.MIN_DISPATCH.value) or equal_gpa(curr_state, TollingState.MAX_DISPATCH)) and \
               ( equal_gpa(new_state, TollingState.MIN_DISPATCH.value) or equal_gpa(new_state, TollingState.MAX_DISPATCH.value))


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
    # cuda section
    if isinstance(can_start, bool):
        return (force_start or is_startup_profitable) if can_start else False

    # GPUArray is can_start
    return gpa_where( invert_bool(can_start)
                    , False
                    , force_start or is_startup_profitable
                    , cuda_ind = True )


def gpa_where( cond       : Union[np.ndarray, GPUArray]
             , cond_true  : Union[float, int, np.short, np.ndarray, GPUArray]
             , cond_false : Union[float, int, np.short, np.ndarray, GPUArray]
             , cuda_ind   : bool = False):
    """ Takes cond_true value if cond is true, and cond_false value if cond is false.

    :param cond: array of boolean values.
    :param cond_true: true value
    :param cond_false: false value.
    :param cuda_ind: indicator for cuda
    :returns: an array taking respective values from cond_true, cond_false

    """
    if not cuda_ind:
        return np.where( cond, cond_true, cond_false )  # this handles bool or array cases

    # CUDA section
    if isinstance(cond, bool):
        return cond_true if cond else cond_false

    if ( not isinstance(cond_true, GPUArray) ) and (not isinstance(cond_false, GPUArray) ):
        return selection_kernel_singles(cond, cond_true, cond_false)

    if (not isinstance(cond_true, GPUArray) ) and isinstance(cond_false, GPUArray):
        return selection_kernel_first_single(cond, cond_true, cond_false)

    if isinstance(cond_true, GPUArray) and (not isinstance(cond_false, GPUArray)):
        return selection_kernel_first_single(invert_bool(cond), cond_false, cond_true)

    # selection kernel for GPUArray
    return selection_kernel( cond, cond_true, cond_false )


def new_state_f(state_state, do_startup, do_shutdown, cuda_ind=False):

    if not cuda_ind:
        return  np.where( (state_state == TollingState.NOT_RUNNING) & do_startup
                        , TollingState.MAX_DISPATCH
                        , np.where( (state_state == TollingState.MAX_DISPATCH) & do_shutdown
                                  , TollingState.NOT_RUNNING
                                  , state_state)
                        )  # remain in the same state

    # cuda version
    state_len = len(state_state)
    max_dispatch = gpa.empty(state_len, dtype=np.short)
    max_dispatch.fill(TollingState.MAX_DISPATCH.value)

    not_running = gpa.empty(state_len, dtype=np.short)
    not_running.fill(TollingState.NOT_RUNNING.value)

    return gpa_where( equal_gpa(state_state, TollingState.NOT_RUNNING.value) and do_startup
                    , max_dispatch
                    , gpa_where( equal_gpa(state_state, TollingState.MAX_DISPATCH.value) and do_shutdown
                               , not_running
                               , state_state  # remain in state
                               , cuda_ind = True)
                    , cuda_ind = True
                    )


def curr_generation_f(new_state, max_cap, min_disp, cuda_ind=False):

    if not cuda_ind:
        return np.where( new_state == TollingState.NOT_RUNNING
                              , 0.
                              , np.where( new_state == TollingState.MAX_DISPATCH
                                        , max_cap
                                        , min_disp ) )

    # cuda result
    return gpa_where( equal_gpa(new_state, TollingState.NOT_RUNNING.value)
                    , gpa.zeros(len(new_state), dtype=float)
                    , gpa_where( equal_gpa(new_state, TollingState.MAX_DISPATCH.value)
                              , max_cap * gpa.ones_like(new_state, dtype=float)
                              , min_disp * gpa.ones_like(new_state, dtype=float)
                              , cuda_ind = True )
                    , cuda_ind = True )  # TODO: OPTIMIZE HERE


def is_cold_start_f(hours_shut, cold_startup_cutoff, cuda_ind = False):
    if not cuda_ind:
        return hours_shut >= cold_startup_cutoff

    if not isinstance(hours_shut, GPUArray):
        return hours_shut >= cold_startup_cutoff

    return bigger_gpa(hours_shut - cold_startup_cutoff)


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
    ramp_rate       = tolling_params['rampRate']
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
    run_at_min_index = (- max_cap) * (power_prices - optimal_marginal_cost_at_max - ramp_up_to_max_cost) + \
                       min_disp * (power_prices - optimal_marginal_cost_at_min - ramp_down_to_min_cost)
    run_at_min_index = run_at_min_index > 0. if not cuda_ind else bigger_gpa(run_at_min_index)

    # compute total startup costs
    is_cold_start     = is_cold_start_f(curr_state['hours_shut'], cold_startup, cuda_ind = cuda_ind)

    fixed_and_fuel_startup_cost = gpa_where( is_cold_start
                                           , fixed_startup_cost_cold + start_fuel_cold * fuel_prices
                                           , fixed_startup_cost + start_fuel * fuel_prices
                                           , cuda_ind = cuda_ind)

    # startup shadow price
    startup_sp = fixed_and_fuel_startup_cost / (startup_horizon * max_cap)  # Startup shadow price TODO: CHECK IF THIS MAKES SENSE
    is_startup_profitable = power_prices - optimal_marginal_cost_at_max - startup_sp
    is_startup_profitable = is_startup_profitable > 0. if not cuda_ind else bigger_gpa(is_startup_profitable)

    do_startup = do_startup_f(curr_decision['can_start'], curr_decision['force_start'], is_startup_profitable, cuda_ind=cuda_ind)

    actual_gen_profit = gpa_where( run_at_min_index
                                 , (power_prices - optimal_marginal_cost_at_min) * min_disp
                                 , (power_prices - optimal_marginal_cost_at_max) * max_cap
                                 , cuda_ind = cuda_ind)

    shutdown_gen_profit = shutdown_horizon * actual_gen_profit
    shut_cost_sp = shutdown_horizon * shutdown_sp_in * max_cap
    # is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp)  # TODO: THIS IS WRONG
    is_shutdown_profitable = - shutdown_gen_profit - (fixed_and_fuel_startup_cost + shut_cost_sp)  # TODO: WRONG
    is_shutdown_profitable = is_shutdown_profitable > 0. if not cuda_ind else bigger_gpa(is_shutdown_profitable)

    # do_shutdown = curr_decision['can_shut'] & ( curr_decision['force_shut'] | is_shutdown_profitable )
    do_shutdown = do_startup_f(curr_decision['can_shut'], curr_decision['force_shut'], is_shutdown_profitable, cuda_ind=cuda_ind)

    # compute new state
    new_state = new_state_f(state_state, do_startup, do_shutdown, cuda_ind=cuda_ind)

    # Generation accounting
    curr_generation = curr_generation_f(new_state, max_cap, min_disp, cuda_ind=cuda_ind)

    generation_change = curr_generation - curr_state['generation']
    # ramping_adjustment = (0.5 / (ramp_rate * hours_in_block)) * np.abs(generation_change) * generation_change
    ramping_adjustment = 0.  # TODO: THIS IS WRONG CHECK HERE
    curr_generation -= ramping_adjustment
    curr_energy = curr_generation * hours_in_block
    revenue = curr_energy * power_prices
    fuel_cost = curr_energy * (fuel_prices + add_fuel_cost) * gpa_where(run_at_min_index, hr_at_min, hr_at_max, cuda_ind=cuda_ind)

    # new starts and new shutdowns.
    starts = starts_indicator(state_state, new_state, cuda_ind=cuda_ind)
    shuts  = shuts_indicator (state_state, new_state, cuda_ind=cuda_ind)

    startup_cost = gpa_where( starts
                           , gpa_where( is_cold_start
                                     , fixed_startup_cost_cold + fuel_prices * start_fuel_cold
                                     , fixed_startup_cost + fuel_prices * start_fuel
                                     , cuda_ind = cuda_ind )
                           , 0. if not cuda_ind else gpa.zeros(len(fuel_prices), dtype=fuel_prices.dtype)
                           , cuda_ind = cuda_ind )

    ramp_cost = gpa_where( (invert_bool(starts) & (generation_change > SMALL_EPS  )) if not cuda_ind else (invert_bool(starts) and bigger_gpa(generation_change - SMALL_EPS ))
                        , ramp_up_cost
                        , gpa_where( (invert_bool(shuts) & (generation_change < - SMALL_EPS)) if not cuda_ind else (invert_bool(shuts) and bigger_gpa(-generation_change - SMALL_EPS))
                                   , ramp_down_cost
                                   , 0.
                                   , cuda_ind = cuda_ind )
                        , cuda_ind = cuda_ind )

    # cashflow = revenue - (totalCost = fuel_cost + (variable_cost = VC * curr_energy) + startup_cost + ramp_cost)
    if not cuda_ind:
        cashflow = np.empty(len(revenue), dtype=float)
        opd_avx.add4(revenue, fuel_cost, VC * curr_energy, startup_cost, ramp_cost, cashflow, nb_paths)
    else:
        cashflow = revenue - fuel_cost - VC * curr_energy - startup_cost - ramp_cost

    # new unit state
    curr_state['hours_in_state']  = gpa_where( new_state == state_state if not cuda_ind else equal_bools(new_state, state_state) # no state change
                                            , curr_state['hours_in_state'] + hours_in_block
                                            , hours_in_block
                                            , cuda_ind = cuda_ind )
    curr_state['generation']      = curr_generation
    curr_state['total_starts']   += starts
    curr_state['hours_shut']      = gpa_where( ((new_state == TollingState.NOT_RUNNING) & (state_state == TollingState.NOT_RUNNING)) if not cuda_ind else
                                               (equal_gpa(new_state, TollingState.NOT_RUNNING.value) and equal_gpa(state_state, TollingState.NOT_RUNNING.value))
                                            , curr_state['hours_shut'] + hours_in_block
                                            , 0
                                            , cuda_ind = cuda_ind )
    curr_state['hours_run']       = gpa_where( still_running_indicator(state_state, new_state, cuda_ind = cuda_ind)
                                             , curr_state['hours_run'] + hours_in_block
                                             , 0
                                             , cuda_ind = cuda_ind )
    curr_state['global_starts']  += starts
    curr_state['state']           = new_state

    return cashflow, curr_state
