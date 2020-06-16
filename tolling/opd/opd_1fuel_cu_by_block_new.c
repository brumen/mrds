#define SMALL_EPS 1e-5
#define DOUBLE_TYPE

#ifdef DOUBLE_TYPE
    #define FLOAT double
#else
    #define FLOAT float
#endif

__global__ void opd_kernel(int    block_nb,
                           FLOAT *power_prices,  /* array of power prices */
                           FLOAT *fuel_prices,  /* array of fuel prices */
                           FLOAT *tolling_params,  /* array of tolling parameters, see below in parameters section */
                           FLOAT *startup_sp,  /* startup shadow price */
                           bool  *state_state,
                           int   *state_hours_in_state,
                           FLOAT *state_generation,
                           int   *state_total_starts,
                           int   *state_hours_shut,
                           int   *state_hours_run,
                           int   *state_global_starts,
                           bool  *dc_can_start,
                           bool  *dc_can_shut,
                           int   *dc_force_start,
                           int   *dc_force_shut,
                           int    hours_in_block,
                           FLOAT  df,   /* discount factor - from day to start of month */
                           int    nb_paths, /* length of vectors, e.g. power, fuel prices, etc. */
                           FLOAT *cashflow  /* cashflow in this block */
                           ) {
  /* Performs a one-block computation of tolling.
     state_... : current state parameters
     dc_...    : parmaeters of the decision criterion.
     cashflow  : vector of cashflow in this block - result of this computation.
   */

  unsigned tid           = threadIdx.x;
  unsigned total_threads = gridDim.x   * blockDim.x;
  unsigned cta_start     = blockDim.x  * blockIdx.x;

  /* parameters */
  FLOAT hr_max            = tolling_params[0];
  FLOAT hr_min            = tolling_params[1];
  FLOAT max_capacity      = tolling_params[2];
  FLOAT min_dispatch      = tolling_params[3];
  FLOAT startup_fuel      = tolling_params[4];
  FLOAT startup_fuel_cold = tolling_params[5];
  FLOAT add_fuel_cost     = tolling_params[6];
  FLOAT variable_cost     = tolling_params[7];
  FLOAT ramp_rate         = tolling_params[8];
  FLOAT shutdown_sp       = tolling_params[9];
  // FLOAT minDownTime    = tolling_params[10];
  // FLOAT minRunTime     = tolling_params[11];
  FLOAT fixed_startup_cost     = tolling_params[12];
  FLOAT fixed_startup_cost_cold = tolling_params[13];
  // FLOAT maxMonthlyStarts  = tolling_params[14];
  FLOAT cold_startup      = tolling_params[15];
  FLOAT startup_horizon   = tolling_params[16];
  FLOAT shutdown_horizon  = tolling_params[17];
  FLOAT rampup_sp         = tolling_params[18];
  FLOAT ramp_down_sp      = tolling_params[19];
  FLOAT ramp_up_cost      = tolling_params[20];
  FLOAT ramp_down_cost    = tolling_params[21];
  FLOAT ramp_up_horizon   = tolling_params[22];
  FLOAT ramp_down_horizon = tolling_params[23];

  for (unsigned idx = cta_start + tid; idx < nb_paths; idx += total_threads) {
      FLOAT fuel_prices_plus_fuel = fuel_prices[idx] + add_fuel_cost;
      FLOAT marginal_cost_at_max  = fuel_prices_plus_fuel * hr_max + variable_cost;
      FLOAT marginal_cost_at_min  = fuel_prices_plus_fuel * hr_min + variable_cost;

      // ramping costs
      bool generation_smaller_maxcap = state_generation[idx] < max_capacity;
#ifdef DOUBLE_TYPE
      FLOAT ramp_up_to_max_cost = generation_smaller_maxcap * (rampup_sp + ramp_up_cost / (max_capacity * ramp_up_horizon));
#else
      FLOAT ramp_up_to_max_cost = generation_smaller_maxcap * (rampup_sp + fdividef(ramp_up_cost, max_capacity * ramp_up_horizon));
#endif
      bool generation_larger_mindisp = state_generation[idx] > min_dispatch;
#ifdef DOUBLE_TYPE
      FLOAT ramp_down_to_min_cost = generation_larger_mindisp * (ramp_down_sp + ramp_down_cost / (min_dispatch * ramp_down_horizon));
#else
      FLOAT ramp_down_to_min_cost = generation_larger_mindisp * (ramp_down_sp + fdividef(ramp_down_cost, min_dispatch * ramp_down_horizon));
#endif

      bool run_at_min_index = max_capacity * (power_prices[idx] - marginal_cost_at_max - ramp_up_to_max_cost) <
        min_dispatch * (power_prices[idx] - marginal_cost_at_min - ramp_down_to_min_cost);
      bool not_run_at_min_index = !run_at_min_index;

      // compute total startup costs
      bool is_cold_start = state_hours_shut[idx] >= cold_startup;
      bool is_not_cold_start = !is_cold_start;
      //  & used to be *
      FLOAT fixed_and_fuel_startup_cost = is_cold_start * (fixed_startup_cost_cold + startup_fuel_cold * fuel_prices[idx]) +
        is_not_cold_start * (fixed_startup_cost + startup_fuel * fuel_prices[idx]);

#ifdef DOUBLE_TYPE
      FLOAT startup_SP = startup_sp[idx] + fixed_and_fuel_startup_cost / (startup_horizon * max_capacity);
#else
      FLOAT startup_SP = startup_sp[idx] + fdividef(fixed_and_fuel_startup_cost, startup_horizon * max_capacity);
#endif

      bool startup_profit_v = power_prices[idx] - marginal_cost_at_max - startup_SP > 0.;
      bool do_startup       = dc_can_start[idx] & ((dc_force_start[idx] == 2) |
                                                (startup_profit_v & (dc_force_start[idx] == 1)));

      // compute shutdown
      FLOAT actual_gen_profit = run_at_min_index * (power_prices[idx] - marginal_cost_at_min) * min_dispatch +
        not_run_at_min_index * (power_prices[idx] - marginal_cost_at_max) * max_capacity;
      FLOAT shutdown_gen_profit = shutdown_horizon * actual_gen_profit;
      FLOAT shut_cost_sp = shutdown_horizon * shutdown_sp * max_capacity;
      bool is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp);
      bool do_shutdown = dc_can_shut[idx] & ((dc_force_shut[idx] == 2) |
                                          (is_shutdown_profitable & (dc_force_shut[idx] == 1)));
      // compute dispatch
      bool not_state_state = !state_state[idx];
      bool curr_state  = (state_state[idx] & (!do_shutdown)) | (not_state_state & do_startup);
      bool state_change    = (curr_state != state_state[idx]);

      // accounting
      FLOAT curr_generation = curr_state * (max_capacity * not_run_at_min_index + run_at_min_index * min_dispatch);
      FLOAT generation_change   = curr_generation - state_generation[idx];

#ifdef DOUBLE_TYPE
      FLOAT ramping_adjustment = (0.5 / ramp_rate / hours_in_block) * abs(generation_change) * generation_change;  // TODO: check the abs here!!!
#else
      FLOAT ramping_adjustment = fdividef(0.5, ramp_rate * hours_in_block) * fabs(generation_change) * generation_change;
#endif

      FLOAT curr_generation  = curr_generation - ramping_adjustment;
      FLOAT curr_energy      = curr_generation * hours_in_block;
      FLOAT revenue          = curr_energy * power_prices[idx];
      FLOAT variable_cost    = variable_cost * curr_energy;
      FLOAT actual_heat_rate = run_at_min_index * hr_min + not_run_at_min_index * hr_max;
      FLOAT fuel_cost        = curr_energy * fuel_prices_plus_fuel * actual_heat_rate;

      bool not_curr_state = !curr_state;
      bool starts         = curr_state & not_state_state;  // curr_state > state_state[idx]
      bool shuts          = not_curr_state & state_state[idx];

      FLOAT startup_cost = (is_cold_start     & starts) * (fixed_startup_cost_cold + fuel_prices[idx] * startup_fuel_cold) +
                                (is_not_cold_start & starts) * (fixed_startup_cost + fuel_prices[idx] * startup_fuel);

      bool ramp_cost_up_ind = (!starts_tmp) & (generation_change > SMALL_EPS);
      bool ramp_cost_dn_ind = (!shuts_tmp) & (generation_change < - SMALL_EPS);
      FLOAT ramp_cost       = ramp_cost_up_ind * ramp_up_cost + ramp_cost_dn_ind * ramp_down_cost;

      // cashflow
      cashflow[idx] = revenue - (fuel_cost + variable_cost + startup_cost + ramp_cost);
      // updating states
      state_hours_in_state[idx] = state_hours_in_state[idx] * (!state_change) + hours_in_block;
      state_generation[idx]     = curr_generation;
      state_total_starts[idx]   = state_total_starts[idx] + starts_tmp;
      state_hours_shut[idx]     = not_curr_state * (state_hours_shut[idx] + hours_in_block);
      state_hours_run[idx]      = curr_state * (state_hours_run[idx] + hours_in_block);
      state_global_starts[idx]  = state_global_starts[idx] + starts_tmp;
      state_state[idx]          = curr_state;

    }
}
