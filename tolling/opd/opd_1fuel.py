import numpy as np
from typing import Tuple

from tolling.opd import opd_avx

SMALL_EPS = 1e-5


def opd_1fuel(power_prices     : np.ndarray
              , fuel_prices    : np.ndarray
              , tolling_params : Tuple
              , startup_sp
              , curr_state
              , curr_decision
              , hours_in_block    : int
              , nb_paths          : int
              , cashflow_per_path : np.ndarray ):
    """ Computes one period tolling optimization.

    :param power_prices: power prices
    :param fuel_prices: fuel prices
    :param tolling_params: parameters of the tolling
    :param startup_sp: startup shadow price.
    :param curr_state: current state vector
    :param curr_decision: current decision vector
    :param hours_in_block: hours for current block
    :param nb_paths: number of paths for , also the length of the power & fuel price vectors.
    :param cashflow_per_path: vector of cashflow for every path
    """

    # unpacking of params
    hr_at_max, hr_at_min, max_cap, min_disp, \
        start_fuel, start_fuel_cold, \
        add_fuel_cost, VC, ramp_rate, \
        shutdown_sp_in, \
        min_downtime, min_runtime, \
        fixed_startup_cost, fixed_startup_cost_cold,  \
        max_monthly_starts, \
        cold_startup, startup_horizon, shutdown_horizon, \
        ramp_up_sp_in, ramp_down_sp_in, \
        ramp_up_cost, ramp_down_cost, ramp_up_horizon, ramp_down_horizon = tolling_params

    state_state = curr_state['state']  # bool type

    # marginal cost at max
    optimal_marginal_cost_at_max = (fuel_prices + add_fuel_cost) * hr_at_max + VC
    optimal_marginal_cost_at_min = (fuel_prices + add_fuel_cost) * hr_at_min + VC

    # ramping costs
    generation_smaller_maxcap = curr_state['generation'] < max_cap
    ramp_up_to_max_cost = generation_smaller_maxcap * (ramp_up_sp_in + ramp_up_cost / (max_cap * ramp_up_horizon))
    generation_larger_mindisp = curr_state['generation'] > min_disp
    ramp_down_to_min_cost = generation_larger_mindisp * (ramp_down_sp_in + ramp_down_cost / min_disp / ramp_down_horizon)
    run_at_min_index = max_cap * (power_prices - optimal_marginal_cost_at_max - ramp_up_to_max_cost) < \
                       min_disp * (power_prices - optimal_marginal_cost_at_min - ramp_down_to_min_cost)
    not_run_at_min_index = ~run_at_min_index

    # compute total startup costs
    is_cold_start     = curr_state['hours_shut'] >= cold_startup
    is_not_cold_start = ~is_cold_start

    fixed_and_fuel_startup_cost = is_cold_start * (fixed_startup_cost_cold + start_fuel_cold * fuel_prices) + \
                                  is_not_cold_start * (fixed_startup_cost + start_fuel * fuel_prices)

    # startup shadow price
    startup_sp += fixed_and_fuel_startup_cost / (startup_horizon * max_cap)  # adjustment of the startup shadow prices TODO: CHECK IF THIS MAKES SENSE
    startup_profit_v = power_prices - optimal_marginal_cost_at_max - startup_sp > 0.

    is_startup_profitable = startup_profit_v
    do_startup = curr_decision['can_start'] * ((curr_decision['force_start'] == 2) |
                                               (is_startup_profitable & (curr_decision['force_start'] == 1)))

    # compute shutdown
    actual_gen_profit = run_at_min_index * (power_prices - optimal_marginal_cost_at_min) * min_disp + \
                        not_run_at_min_index * (power_prices - optimal_marginal_cost_at_max) * max_cap
    shutdown_gen_profit = shutdown_horizon * actual_gen_profit
    shut_cost_sp = shutdown_horizon * shutdown_sp_in * max_cap
    is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp)
    do_shutdown = curr_decision['can_shut'] & ((curr_decision['force_shut'] == 2) |
                                               (is_shutdown_profitable & (curr_decision['force_shut'] == 1)))
    # compute dispatch
    not_state_state = ~state_state
    curr_state      = (state_state & (~do_shutdown)) | (not_state_state & do_startup)
    state_change    = curr_state != state_state

    # accounting (- on curr_state is because curr_state is -1)
    curr_generation = curr_state * (max_cap * not_run_at_min_index + run_at_min_index * min_disp)
    generation_change = curr_generation - curr_state['generation']
    ramping_adjustment = (0.5 / (ramp_rate * hours_in_block)) * np.abs(generation_change) * generation_change
    curr_generation -= ramping_adjustment
    curr_energy = curr_generation * hours_in_block
    revenue = curr_energy * power_prices
    variable_cost = VC * curr_energy
    actual_heat_rate = run_at_min_index * hr_at_min + not_run_at_min_index * hr_at_max
    fuel_cost = curr_energy * (fuel_prices + add_fuel_cost) * actual_heat_rate

    not_curr_state = ~curr_state
    starts = curr_state & not_state_state  # curr_state > state_state
    shuts = not_curr_state & state_state

    startup_cost = (is_cold_start & starts) * (fixed_startup_cost_cold + fuel_prices * start_fuel_cold) + \
                   (is_not_cold_start & starts) * (fixed_startup_cost + fuel_prices * start_fuel)

    ramp_cost = (~starts) & (generation_change > SMALL_EPS  ) * ramp_up_cost + \
                (~shuts ) & (generation_change < - SMALL_EPS) * ramp_down_cost

    # this is replaced by the opd_avx function
    # totalCost = fuel_cost + variable_cost + startup_cost + ramp_cost
    # cashflow = revenue - totalCost
    # cashflow_per_path[:] = cashflow
    opd_avx.add4(revenue, fuel_cost, variable_cost, startup_cost, ramp_cost, cashflow_per_path, nb_paths)

    # new unit state
    curr_state['hours_in_state']  = curr_state['hours_in_state'] * (~state_change) + hours_in_block
    curr_state['generation']      = curr_generation
    curr_state['total_starts']   += starts
    curr_state['hours_shut']      = not_curr_state * (curr_state['hours_shut'] + hours_in_block)
    curr_state['hours_run']       = curr_state * (curr_state['hours_run'] + hours_in_block)
    curr_state['global_starts']  += starts
    curr_state['state']           = curr_state
