// goes through the month and computes the revenue.

#define FLOAT float  // float type

#define DOUBLE_USED

nus_hours_in_state = 0;
nus_generation     = 0.;
nus_total_starts   = 0;
nus_hours_shut     = 0;
nus_hours_run      = 0;
nus_global_starts  = 0;
nus_state          = 0;   // TODO: CHECK HERE !!!!


for (block_nb = 0; block_nb < block_nbs; block_nb += 1) {

  FLOAT fp_plus_fuel = fp[i] + addFuelCost;
  FLOAT optimal_marginal_cost_at_max = fp_plus_fuel * hrAtMax + VC;
  FLOAT optimal_marginal_cost_at_min = fp_plus_fuel * hrAtMin + VC;

  // ramping costs
  bool generation_smaller_maxcap = state_Generation[i] < maxCap;
  #ifdef DOUBLE_USED
    FLOAT ramp_up_to_max_cost = generation_smaller_maxcap * (rampUpSPin + rampUpCost / (maxCap * rampUpHorizon));
  #else
    FLOAT ramp_up_to_max_cost = generation_smaller_maxcap * (rampUpSPin + fdividef(rampUpCost, maxCap * rampUpHorizon));
  #endif
  bool generation_larger_mindisp = state_Generation[i] > minDisp;

  #ifdef DOUBLE_USED
    FLOAT ramp_down_to_min_cost = generation_larger_mindisp * (rampDownSPin + rampDownCost / (minDisp * rampDownHorizon));
  #else
    FLOAT ramp_down_to_min_cost = generation_larger_mindisp * (rampDownSPin + fdividef(rampDownCost, minDisp * rampDownHorizon));
  #endif

  bool run_at_min_index = maxCap * (pp[i] - optimal_marginal_cost_at_max - ramp_up_to_max_cost) <
    minDisp * (pp[i] - optimal_marginal_cost_at_min - ramp_down_to_min_cost);
  bool not_run_at_min_index = !run_at_min_index;

  // compute total startup costs
  bool is_cold_start     = state_hoursShut[i] >= coldStartup;
  bool is_not_cold_start = !is_cold_start;
  //  & used to be *
  FLOAT fixed_and_fuel_startup_cost = is_cold_start * (fixedStartupCostCold + startFuelCold * fp[i]) +
    is_not_cold_start * (fixedStartupCost + startFuel * fp[i]);

  #ifdef DOUBLE_USED
    FLOAT startup_SP = startupSPin[i] + fixed_and_fuel_startup_cost / (startupHorizon * maxCap);
  #else
    FLOAT startup_SP = startupSPin[i] + fdividef(fixed_and_fuel_startup_cost, startupHorizon * maxCap);
  #endif
  bool startup_profit_v = pp[i] - optimal_marginal_cost_at_max - startup_SP > 0.;
  bool do_startup       = dc_canStart[i] * ((dc_forceStart[i] == 2) | (startup_profit_v & (dc_forceStart[i] == 1)));

  // compute shutdown
  FLOAT actual_gen_profit = run_at_min_index * (pp[i] - optimal_marginal_cost_at_min) * minDisp +
    not_run_at_min_index * (pp[i] - optimal_marginal_cost_at_max) * maxCap;
  FLOAT shutdown_gen_profit = shutdownHorizon * actual_gen_profit;
  FLOAT shut_cost_sp = shutdownHorizon * shutdownSPin * maxCap;
  bool is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp);
  bool do_shutdown = dc_canShut[i] & ((dc_forceShut[i] == 2) |
                                      (is_shutdown_profitable & (dc_forceShut[i] == 1)));
  // compute dispatch
  bool not_state_state = !state_state[i];
  bool curr_state_tmp = (state_state[i] & (!do_shutdown)) | (not_state_state & do_startup);
  curr_state[i] = curr_state_tmp;
  bool state_change = (curr_state_tmp != state_state[i]);

  // accounting
  FLOAT curr_generation_tmp = curr_state_tmp * (maxCap * not_run_at_min_index + run_at_min_index * minDisp);
  FLOAT generation_change = curr_generation_tmp - state_Generation[i];
  #ifdef DOUBLE_USED
    FLOAT ramping_adjustment = (0.5 / rampRate / hours_in_block) * abs(generation_change) * generation_change;  // TODO: check the abs here!!!
  #else
    FLOAT ramping_adjustment = fdividef(0.5, rampRate * hours_in_block) * fabs(generation_change) * generation_change;
  #endif
  FLOAT curr_generation  = curr_generation_tmp - ramping_adjustment;
  FLOAT curr_energy      = curr_generation * hours_in_block;
  FLOAT revenue          = curr_energy * pp[i];
  FLOAT variable_cost    = VC * curr_energy;
  FLOAT actual_heat_rate = run_at_min_index * hrAtMin + not_run_at_min_index * hrAtMax;
  FLOAT fuel_cost        = curr_energy * fp_plus_fuel * actual_heat_rate;

  bool not_curr_state = !curr_state_tmp;
  bool starts_tmp = curr_state_tmp & not_state_state;  // curr_state > state_state[i]
  shuts_tmp = not_curr_state & state_state[i];

  FLOAT startup_cost = (is_cold_start & starts_tmp) * (fixedStartupCostCold + fp[i] * startFuelCold) +
    (is_not_cold_start & starts_tmp) * (fixedStartupCost + fp[i] * startFuel);

  bool ramp_cost_up_ind = (!starts_tmp) & (generation_change > SMALL_EPS);
  bool ramp_cost_dn_ind = (!shuts_tmp) & (generation_change < - SMALL_EPS);
  FLOAT ramp_cost = ramp_cost_up_ind * rampUpCost + ramp_cost_dn_ind * rampDownCost;
  // cashflow
  cashflow[i] = revenue - (fuel_cost + variable_cost + startup_cost + ramp_cost);

  // new unit state
  nus_hours_in_state[i] = state_hoursInState[i] * (!state_change) + hours_in_block;
  nus_generation[i]     = curr_generation_tmp;
  nus_total_starts[i]   = state_TotalStarts[i] + starts_tmp;
  nus_hours_shut[i]     = not_curr_state * (state_hoursShut[i] + hours_in_block);
  nus_hours_run[i]      = curr_state_tmp * (state_hoursRun[i] + hours_in_block);
  nus_global_starts[i]  = state_globalStarts[i] + starts_tmp;
  nus_state[i]          = curr_state_tmp;

 }
