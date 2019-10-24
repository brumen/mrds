#define SMALL_EPS 1e-5
#define DOUBLE_TYPE

#ifdef DOUBLE_TYPE
    #define FLOAT_TYPE double
#else
    #define FLOAT_TYPE float
#endif

__global__ void opd_kernel(int     block_nb,
                           double *powerPrice,
                           double *fuelPrice,
                           double *opdParams,
                           double *startupSPin,
                           bool   *state_state,
                           int    *state_hoursInState,
                           double *state_Generation,
                           int    *state_TotalStarts,
                           int    *state_hoursShut,
                           int    *state_hoursRun,
                           int    *state_globalStarts,
                           bool   *dc_canStart,
                           bool   *dc_canShut,
                           int    *dc_forceStart,
                           int    *dc_forceShut,
                           int     hours_in_block,
                           double  df,
                           int     nb_paths,
                           unsigned long n) {

  unsigned tid = threadIdx.x;
  unsigned total_threads = gridDim.x*blockDim.x;
  unsigned cta_start = blockDim.x * blockIdx.x;
  unsigned i;

  // parameters
  double hrAtMax        = opdParams[0];
  double hrAtMin        = opdParams[1];
  double maxCap         = opdParams[2];
  double minDisp        = opdParams[3];
  double startFuel      = opdParams[4];
  double startFuelCold  = opdParams[5];
  double addFuelCost    = opdParams[6];
  double VC                   = opdParams[7];
  double rampRate             = opdParams[8];
  double shutdownSPin         = opdParams[9];
  // double minDownTime       = opdParams[10];
  // double minRunTime        = opdParams[11];
  double fixedStartupCost     = opdParams[12];
  double fixedStartupCostCold = opdParams[13];
  // double maxMonthlyStarts  = opdParams[14];
  double coldStartup     = opdParams[15];
  double startupHorizon  = opdParams[16];
  double shutdownHorizon = opdParams[17];
  double rampUpSPin      = opdParams[18];
  double rampDownSPin    = opdParams[19];
  double rampUpCost      = opdParams[20];
  double rampDownCost    = opdParams[21];
  double rampUpHorizon   = opdParams[22];
  double rampDownHorizon = opdParams[23];

  for (i = cta_start + tid; i < n; i += total_threads) {
      // computes one period optimization,
      FLOAT_TYPE fuelPrice_plus_fuel          = fuelPrice[i] + addFuelCost;
      FLOAT_TYPE optimal_marginal_cost_at_max = fuelPrice_plus_fuel * hrAtMax + VC;
      FLOAT_TYPE optimal_marginal_cost_at_min = fuelPrice_plus_fuel * hrAtMin + VC;

      // ramping costs
      bool generation_smaller_maxcap = state_Generation[i] < maxCap;
#ifdef DOUBLE_TYPE
      FLOAT_TYPE ramp_up_to_max_cost = generation_smaller_maxcap * (rampUpSPin + rampUpCost / (maxCap * rampUpHorizon));
#else
      FLOAT_TYPE ramp_up_to_max_cost = generation_smaller_maxcap * (rampUpSPin + fdividef(rampUpCost, maxCap * rampUpHorizon));
#endif
      bool generation_larger_mindisp = state_Generation[i] > minDisp;
#ifdef DOUBLE_TYPE
      FLOAT_TYPE ramp_down_to_min_cost = generation_larger_mindisp * (rampDownSPin + rampDownCost / (minDisp * rampDownHorizon));
#else
      FLOAT_TYPE ramp_down_to_min_cost = generation_larger_mindisp * (rampDownSPin + fdividef(rampDownCost, minDisp * rampDownHorizon));
#endif

      bool run_at_min_index = maxCap * (powerPrice[i] - optimal_marginal_cost_at_max - ramp_up_to_max_cost) <
        minDisp * (powerPrice[i] - optimal_marginal_cost_at_min - ramp_down_to_min_cost);
      bool not_run_at_min_index = !run_at_min_index;

      // compute total startup costs
      bool is_cold_start = state_hoursShut[i] >= coldStartup;
      bool is_not_cold_start = !is_cold_start;
      //  & used to be *
      FLOAT_TYPE fixed_and_fuel_startup_cost = is_cold_start * (fixedStartupCostCold + startFuelCold * fuelPrice[i]) +
        is_not_cold_start * (fixedStartupCost + startFuel * fuelPrice[i]);

#ifdef DOUBLE_TYPE
      FLOAT_TYPE startup_SP = startupSPin[i] + fixed_and_fuel_startup_cost / (startupHorizon * maxCap);
#else
      FLOAT_TYPE startup_SP = startupSPin[i] + fdividef(fixed_and_fuel_startup_cost, startupHorizon * maxCap);
#endif

      bool startup_profit_v = powerPrice[i] - optimal_marginal_cost_at_max - startup_SP > 0.;
      bool do_startup = dc_canStart[i] * ((dc_forceStart[i] == 2) |
                                          (startup_profit_v & (dc_forceStart[i] == 1)));

      // compute shutdown
      FLOAT_TYPE actual_gen_profit = run_at_min_index * (powerPrice[i] - optimal_marginal_cost_at_min) * minDisp +
        not_run_at_min_index * (powerPrice[i] - optimal_marginal_cost_at_max) * maxCap;
      FLOAT_TYPE shutdown_gen_profit = shutdownHorizon * actual_gen_profit;
      FLOAT_TYPE shut_cost_sp = shutdownHorizon * shutdownSPin * maxCap;
      bool is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp);
      bool do_shutdown = dc_canShut[i] & ((dc_forceShut[i] == 2) |
                                          (is_shutdown_profitable & (dc_forceShut[i] == 1)));
      bool gorazd = true;
      // compute dispatch
      bool not_state_state = !state_state[i];
      bool curr_state_tmp  = (state_state[i] & (!do_shutdown)) | (not_state_state & do_startup);
      bool state_change    = (curr_state_tmp != state_state[i]);

      // accounting
      FLOAT_TYPE curr_generation_tmp = curr_state_tmp * (maxCap * not_run_at_min_index + run_at_min_index * minDisp);
      FLOAT_TYPE generation_change   = curr_generation_tmp - state_Generation[i];

#ifdef DOUBLE_TYPE
      FLOAT_TYPE ramping_adjustment = (0.5 / rampRate / hours_in_block) * abs(generation_change) * generation_change;  // TODO: check the abs here!!!
#else
      FLOAT_TYPE ramping_adjustment = fdividef(0.5, rampRate * hours_in_block) * fabs(generation_change) * generation_change;
#endif

      FLOAT_TYPE curr_generation = curr_generation_tmp - ramping_adjustment;
      FLOAT_TYPE curr_energy = curr_generation * hours_in_block;
      FLOAT_TYPE revenue = curr_energy * powerPrice[i];
      FLOAT_TYPE variable_cost = VC * curr_energy;
      FLOAT_TYPE actual_heat_rate = run_at_min_index * hrAtMin + not_run_at_min_index * hrAtMax;
      FLOAT_TYPE fuel_cost = curr_energy * fuelPrice_plus_fuel * actual_heat_rate;

      bool not_curr_state = !curr_state_tmp;
      bool starts_tmp     = curr_state_tmp & not_state_state;  // curr_state > state_state[i]
      bool shuts_tmp      = not_curr_state & state_state[i];

      //FLOAT_TYPE startup_cost = (is_cold_start & starts_tmp) * (fixedStartupCostCold + fuelPrice[i] * startFuelCold) +
      //  (is_not_cold_start & starts_tmp) * (fixedStartupCost + fuelPrice[i] * startFuel);

      //bool ramp_cost_up_ind = (!starts_tmp) & (generation_change > SMALL_EPS);
      //bool ramp_cost_dn_ind = (!shuts_tmp) & (generation_change < - SMALL_EPS);
      // FLOAT_TYPE ramp_cost = ramp_cost_up_ind * rampUpCost + ramp_cost_dn_ind * rampDownCost;
      // cashflow
      // FLOAT_TYPE cashflow = revenue - (fuel_cost + variable_cost + startup_cost + ramp_cost);

      // new states
      state_hoursInState[i] = state_hoursInState[i] * (!state_change) + hours_in_block;
      state_Generation[i]   = curr_generation_tmp;
      state_TotalStarts[i]  = state_TotalStarts[i] + starts_tmp;
      state_hoursShut[i]    = not_curr_state * (state_hoursShut[i] + hours_in_block);
      state_hoursRun[i]     = curr_state_tmp * (state_hoursRun[i] + hours_in_block);
      state_globalStarts[i] = state_globalStarts[i] + starts_tmp;
      state_state[i]        = curr_state_tmp;

    }
}
