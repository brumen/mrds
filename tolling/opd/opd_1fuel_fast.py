# one period tolling dispatch

import numpy as np

from tolling.opd import add4

SMALL_EPS = 1e-5


def opd_1fuel_fast( power_prices  # vector of power prices
                  , fuel_prices   # vector of fuel prices
                   # scalars for 1 fuel
                    , hr_max
                    , hr_min
                    , max_cap
                    , min_disp  # max, min dispatch
                    , start_fuel
                    , start_fuel_cold  # scalar
                    , added_fuel_cost,  # scalar
                   VC,  # variable cost, scalar
                   ramp_rate,  # scalar
                   startup_sp_in,
                   shutdown_sp_in,
                   startup_cost,
                   startup_cost_cold,
                   xud_cold_startup,
                   xud_startup_horizon,
                   xud_shutdown_horizon,
                   xud_ramp_up_sp_in,
                   xud_ramp_down_sp_in,
                   xud_ramp_up_cost,
                   xud_ramp_down_cost,
                   xud_ramp_up_horizon,
                   xud_ramp_down_horizon,
                   # vectors of current state
                   xus_state,  # vector
                   xus_hours_in_state,  # vector
                   xus_generation,  # vector
                   xus_total_starts,  # vector
                   xus_hours_shut,  # vector
                   xus_hours_run,  # vector
                   xus_global_starts,  # vector
                   # dispatch constraints
                   dc_can_start,
                   dc_can_shut,
                   dc_force_start,
                   dc_force_shut,
                   hours_in_block,
                   nb_paths,
                   # new unit state
                   nus_hours_in_state,
                   nus_generation,
                   nus_total_starts,
                   nus_hours_shut,
                   nus_hours_run,
                   nus_global_starts,
                   nus_state,
                   cf_per_path,
                   tmp_vars):
    """
    computes one period optimization,
    :param power_prices: power prices (on device)
    :param fuel_prices: fuel prices (on device)
    :param cuda_ind: indicator whether to do the computation on CUDA
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
        ramp_up_cost, ramp_down_cost, ramp_up_horizon, ramp_down_horizon = params

    # marginal cost at max
    optimal_marginal_cost_at_max = (fuel_prices + add_fuel_cost) * hr_at_max + VC
    optimal_marginal_cost_at_min = (fuel_prices + add_fuel_cost) * hr_at_min + VC

    # ramping costs
    generation_smaller_maxcap = state_generation < max_cap
    ramp_up_to_max_cost = generation_smaller_maxcap * (ramp_up_sp_in + ramp_up_cost / (max_cap * ramp_up_horizon))
    generation_larger_mindisp = state_generation > min_disp
    ramp_down_to_min_cost = generation_larger_mindisp * (ramp_down_sp_in + ramp_down_cost / min_disp / ramp_down_horizon)
    run_at_min_index = max_cap * (power_prices - optimal_marginal_cost_at_max - ramp_up_to_max_cost) < \
        min_disp * (power_prices - optimal_marginal_cost_at_min - ramp_down_to_min_cost)
    not_run_at_min_index = ~run_at_min_index

    # compute total startup costs
    is_cold_start = state_hours_shut >= cold_startup
    is_not_cold_start = ~is_cold_start

    fixed_and_fuel_startup_cost = is_cold_start * (fixed_startup_cost_cold + start_fuel_cold * fp) + \
        is_not_cold_start * (fixed_startup_cost + start_fuel * fp)

    startup_sp = startup_sp_in + fixed_and_fuel_startup_cost / (startup_horizon * max_cap)
    is_startup_profitable = power_prices - optimal_marginal_cost_at_max - startup_sp > 0.

    do_startup = dc_can_start * ((dc_force_start == 2) |
                                (is_startup_profitable & (dc_force_start == 1)))

    # compute shutdown
    actual_gen_profit = run_at_min_index * (power_prices - optimal_marginal_cost_at_min) * min_disp + \
        not_run_at_min_index * (power_prices - optimal_marginal_cost_at_max) * max_cap
    shutdown_gen_profit = shutdown_horizon * actual_gen_profit
    shut_cost_sp = shutdown_horizon * shutdown_sp_in * max_cap
    is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp)
    do_shutdown = dc_can_shut & ((dc_force_shut == 2) |
                                 (is_shutdown_profitable & (dc_force_shut == 1)))
    # compute dispatch
    not_state_state = ~state_state
    curr_state      = (state_state & (~do_shutdown)) | (not_state_state & do_startup)
    state_change    = curr_state != state_state

    # accounting (- on curr_state is because curr_state is -1)
    curr_generation    = curr_state * (max_cap * not_run_at_min_index + run_at_min_index * min_disp)
    generation_change  = curr_generation - state_generation
    ramping_adjustment = (0.5 / (ramp_rate * hours_in_block)) * np.abs(generation_change) * generation_change
    curr_generation -= ramping_adjustment
    curr_energy = curr_generation * hours_in_block
    revenue = curr_energy * power_prices
    variable_cost = VC * curr_energy
    actual_heat_rate = run_at_min_index * hr_at_min + not_run_at_min_index * hr_at_max

    not_curr_state = ~curr_state
    starts = curr_state & not_state_state  # curr_state > state_state
    shuts  = not_curr_state & state_state

    startup_cost = (is_cold_start & starts)     * (fixed_startup_cost_cold + fuel_prices * start_fuel_cold) + \
                   (is_not_cold_start & starts) * (fixed_startup_cost + fuel_prices * start_fuel)

    ramp_cost_up_ind = (~starts) & (generation_change > SMALL_EPS)
    ramp_cost_dn_ind = (~shuts) & (generation_change < - SMALL_EPS)
    ramp_cost = ramp_cost_up_ind * ramp_up_cost + ramp_cost_dn_ind * ramp_down_cost

    # this is replaced by the opd_avx function
    # totalCost = fuel_cost + variable_cost + startup_cost + ramp_cost
    # cashflow = revenue - totalCost
    # cashflow[:] = cashflow
    add4( revenue
        , curr_energy * (fuel_prices + add_fuel_cost) * actual_heat_rate  # fuel costs
        , variable_cost
        , startup_cost
        , ramp_cost
        , cashflow_per_path
        , nb_paths )

    # new unit state
    nus_hours_in_state[:] = state_hours_in_state * (~state_change) + hours_in_block
    nus_generation[:]     = curr_generation
    nus_total_starts[:]   = state_total_starts + starts
    nus_hours_shut[:]     = not_curr_state * (state_hours_shut + hours_in_block)
    nus_hours_run[:]      = curr_state * (state_hours_run + hours_in_block)
    nus_global_starts[:]  = state_global_starts + starts
    nus_state[:]          = curr_state
