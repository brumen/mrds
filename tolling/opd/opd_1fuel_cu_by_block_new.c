#define SMALL_EPS 1e-5
#define DOUBLE_TYPE

#ifdef DOUBLE_TYPE
    #define FLOAT_TYPE double
#else
    #define FLOAT_TYPE float
#endif

__global__ void opd_kernel(int         block_nb,
                           FLOAT_TYPE *power_prices,  /* array of power prices */
                           FLOAT_TYPE *fuel_prices,  /* array of fuel prices */
                           FLOAT_TYPE *tolling_params,  /* array of tolling parameters, see below in parameters section */
                           FLOAT_TYPE *startup_sp,  /* startup shadow price */
                           bool       *state_state,
                           int        *state_hours_in_state,
                           FLOAT_TYPE *state_generation,
                           int        *state_total_starts,
                           int        *state_hours_shut,
                           int        *state_hours_run,
                           int        *state_global_starts,
                           bool       *dc_can_start,
                           bool       *dc_can_shut,
                           int        *dc_force_start,
                           int        *dc_force_shut,
                           int         hours_in_block,
                           FLOAT_TYPE  df,   /* discount factor - from day to start of month */
                           int         nb_paths, /* length of vectors, e.g. power, fuel prices, etc. */
                           FLOAT_TYPE *cashflow  /* cashflow in this block */
                           ) {
  /* Performs a one-block computation of tolling.
     state_... : parameters of the state
     dc_...    : parmaeters of the decision criterion.

     cashflow: vector of cashflow in this block - result of this computation.
   */

  unsigned tid = threadIdx.x;
  unsigned total_threads = gridDim.x*blockDim.x;
  unsigned cta_start = blockDim.x * blockIdx.x;
  unsigned i;

  /* parameters */
  FLOAT_TYPE hr_max        = tolling_params[0];
  FLOAT_TYPE hr_min        = tolling_params[1];
  FLOAT_TYPE max_capacity  = tolling_params[2];
  FLOAT_TYPE min_dispatch  = tolling_params[3];
  FLOAT_TYPE startup_fuel      = tolling_params[4];
  FLOAT_TYPE startup_fuel_cold  = tolling_params[5];
  FLOAT_TYPE add_fuel_cost    = tolling_params[6];
  FLOAT_TYPE variable_cost                   = tolling_params[7];
  FLOAT_TYPE rampRate             = tolling_params[8];
  FLOAT_TYPE shutdownSPin         = tolling_params[9];
  // FLOAT_TYPE minDownTime       = tolling_params[10];
  // FLOAT_TYPE minRunTime        = tolling_params[11];
  FLOAT_TYPE fixedStartupCost     = tolling_params[12];
  FLOAT_TYPE fixedStartupCostCold = tolling_params[13];
  // FLOAT_TYPE maxMonthlyStarts  = tolling_params[14];
  FLOAT_TYPE coldStartup     = tolling_params[15];
  FLOAT_TYPE startupHorizon  = tolling_params[16];
  FLOAT_TYPE shutdownHorizon = tolling_params[17];
  FLOAT_TYPE rampUpSPin      = tolling_params[18];
  FLOAT_TYPE rampDownSPin    = tolling_params[19];
  FLOAT_TYPE rampUpCost      = tolling_params[20];
  FLOAT_TYPE rampDownCost    = tolling_params[21];
  FLOAT_TYPE rampUpHorizon   = tolling_params[22];
  FLOAT_TYPE rampDownHorizon = tolling_params[23];

  for (i = cta_start + tid; i < nb_paths; i += total_threads) {
      FLOAT_TYPE fuel_prices_plus_fuel = fuel_prices[i] + add_fuel_cost;
      FLOAT_TYPE marginal_cost_at_max  = fuel_prices_plus_fuel * hr_max + variable_cost;
      FLOAT_TYPE marginal_cost_at_min  = fuel_prices_plus_fuel * hr_min + variable_cost;

      // ramping costs
      bool generation_smaller_maxcap = state_generation[i] < max_capacity;
#ifdef DOUBLE_TYPE
      FLOAT_TYPE ramp_up_to_max_cost = generation_smaller_maxcap * (rampUpSPin + rampUpCost / (max_capacity * rampUpHorizon));
#else
      FLOAT_TYPE ramp_up_to_max_cost = generation_smaller_maxcap * (rampUpSPin + fdividef(rampUpCost, max_capacity * rampUpHorizon));
#endif
      bool generation_larger_mindisp = state_generation[i] > min_dispatch;
#ifdef DOUBLE_TYPE
      FLOAT_TYPE ramp_down_to_min_cost = generation_larger_mindisp * (rampDownSPin + rampDownCost / (min_dispatch * rampDownHorizon));
#else
      FLOAT_TYPE ramp_down_to_min_cost = generation_larger_mindisp * (rampDownSPin + fdividef(rampDownCost, min_dispatch * rampDownHorizon));
#endif

      bool run_at_min_index = max_capacity * (power_prices[i] - marginal_cost_at_max - ramp_up_to_max_cost) <
        min_dispatch * (power_prices[i] - marginal_cost_at_min - ramp_down_to_min_cost);
      bool not_run_at_min_index = !run_at_min_index;

      // compute total startup costs
      bool is_cold_start = state_hours_shut[i] >= coldStartup;
      bool is_not_cold_start = !is_cold_start;
      //  & used to be *
      FLOAT_TYPE fixed_and_fuel_startup_cost = is_cold_start * (fixedStartupCostCold + startup_fuel_cold * fuel_prices[i]) +
        is_not_cold_start * (fixedStartupCost + startup_fuel * fuel_prices[i]);

#ifdef DOUBLE_TYPE
      FLOAT_TYPE startup_SP = startup_sp[i] + fixed_and_fuel_startup_cost / (startupHorizon * max_capacity);
#else
      FLOAT_TYPE startup_SP = startup_sp[i] + fdividef(fixed_and_fuel_startup_cost, startupHorizon * max_capacity);
#endif

      bool startup_profit_v = power_prices[i] - marginal_cost_at_max - startup_SP > 0.;
      bool do_startup = dc_can_start[i] * ((dc_force_start[i] == 2) |
                                          (startup_profit_v & (dc_force_start[i] == 1)));

      // compute shutdown
      FLOAT_TYPE actual_gen_profit = run_at_min_index * (power_prices[i] - marginal_cost_at_min) * min_dispatch +
        not_run_at_min_index * (power_prices[i] - marginal_cost_at_max) * max_capacity;
      FLOAT_TYPE shutdown_gen_profit = shutdownHorizon * actual_gen_profit;
      FLOAT_TYPE shut_cost_sp = shutdownHorizon * shutdownSPin * max_capacity;
      bool is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp);
      bool do_shutdown = dc_can_shut[i] & ((dc_force_shut[i] == 2) |
                                          (is_shutdown_profitable & (dc_force_shut[i] == 1)));
      // compute dispatch
      bool not_state_state = !state_state[i];
      bool curr_state_tmp  = (state_state[i] & (!do_shutdown)) | (not_state_state & do_startup);
      bool state_change    = (curr_state_tmp != state_state[i]);

      // accounting
      FLOAT_TYPE curr_generation_tmp = curr_state_tmp * (max_capacity * not_run_at_min_index + run_at_min_index * min_dispatch);
      FLOAT_TYPE generation_change   = curr_generation_tmp - state_generation[i];

#ifdef DOUBLE_TYPE
      FLOAT_TYPE ramping_adjustment = (0.5 / rampRate / hours_in_block) * abs(generation_change) * generation_change;  // TODO: check the abs here!!!
#else
      FLOAT_TYPE ramping_adjustment = fdividef(0.5, rampRate * hours_in_block) * fabs(generation_change) * generation_change;
#endif

      FLOAT_TYPE curr_generation  = curr_generation_tmp - ramping_adjustment;
      FLOAT_TYPE curr_energy      = curr_generation * hours_in_block;
      FLOAT_TYPE revenue          = curr_energy * power_prices[i];
      FLOAT_TYPE variable_cost    = variable_cost * curr_energy;
      FLOAT_TYPE actual_heat_rate = run_at_min_index * hr_min + not_run_at_min_index * hr_max;
      FLOAT_TYPE fuel_cost        = curr_energy * fuel_prices_plus_fuel * actual_heat_rate;

      bool not_curr_state = !curr_state_tmp;
      bool starts_tmp     = curr_state_tmp & not_state_state;  // curr_state > state_state[i]
      bool shuts_tmp      = not_curr_state & state_state[i];

      FLOAT_TYPE startup_cost = (is_cold_start & starts_tmp) * (fixedStartupCostCold + fuel_prices[i] * startup_fuel_cold) +
                                (is_not_cold_start & starts_tmp) * (fixedStartupCost + fuel_prices[i] * startup_fuel);

      bool ramp_cost_up_ind = (!starts_tmp) & (generation_change > SMALL_EPS);
      bool ramp_cost_dn_ind = (!shuts_tmp) & (generation_change < - SMALL_EPS);
      FLOAT_TYPE ramp_cost = ramp_cost_up_ind * rampUpCost + ramp_cost_dn_ind * rampDownCost;

      // cashflow
      cashflow[i] = revenue - (fuel_cost + variable_cost + startup_cost + ramp_cost);
      // new states
      state_hours_in_state[i] = state_hours_in_state[i] * (!state_change) + hours_in_block;
      state_generation[i]   = curr_generation_tmp;
      state_total_starts[i]  = state_total_starts[i] + starts_tmp;
      state_hours_shut[i]    = not_curr_state * (state_hours_shut[i] + hours_in_block);
      state_hours_run[i]     = curr_state_tmp * (state_hours_run[i] + hours_in_block);
      state_global_starts[i] = state_global_starts[i] + starts_tmp;
      state_state[i]        = curr_state_tmp;

    }
}
