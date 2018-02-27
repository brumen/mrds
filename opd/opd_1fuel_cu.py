# elementwise kernel for one_period dispatch
import config
import pycuda.gpuarray as gpa
from pycuda.elementwise import ElementwiseKernel
import pycuda.compiler


inputs = """
    int block_nb,
    float *pp, float *fp,
    float *params_cuda,
    float *startupSPin,
    bool  *state_state,
    int *state_hoursInState,
    float *state_Generation,
    int *state_TotalStarts,
    int *state_hoursShut,
    int *state_hoursRun,
    int *state_globalStarts,
    bool  *dc_canStart,
    bool  *dc_canShut,
    int   *dc_forceStart,
    int   *dc_forceShut,
    int   hours_in_block,
    float df,
    int   nb_paths,
    int   *nus_hours_in_state,
    float *nus_generation,
    int   *nus_total_starts,
    int   *nus_hours_shut,
    int   *nus_hours_run,
    int   *nus_global_starts,
    bool  *nus_state,
    float *cashflow,
    bool *curr_state
"""

loop_prep = """
    // parameters 
    float hrAtMax = params_cuda[0];
    float hrAtMin  = params_cuda[1];
    float maxCap  = params_cuda[2];
    float minDisp  = params_cuda[3];
    float startFuel  = params_cuda[4];
    float startFuelCold  = params_cuda[5];
    float addFuelCost = params_cuda[6];
    float VC = params_cuda[7];
    float rampRate = params_cuda[8];
    float shutdownSPin = params_cuda[9];
    // float minDownTime = params_cuda[10];
    // float minRunTime = params_cuda[11];
    float fixedStartupCost = params_cuda[12];
    float fixedStartupCostCold = params_cuda[13];
    // float maxMonthlyStarts = params_cuda[14];
    float coldStartup = params_cuda[15];
    float startupHorizon = params_cuda[16];
    float shutdownHorizon = params_cuda[17];
    float rampUpSPin = params_cuda[18];
    float rampDownSPin = params_cuda[19];
    float rampUpCost = params_cuda[20];
    float rampDownCost = params_cuda[21];
    float rampUpHorizon = params_cuda[22];
    float rampDownHorizon = params_cuda[23];
    bool  shuts_tmp;
"""

loop_code = """
    // computes one period optimization,
    // marginal cost at max
    float fp_plus_fuel = fp[i] + addFuelCost;
    float optimal_marginal_cost_at_max = fp_plus_fuel * hrAtMax + VC;
    float optimal_marginal_cost_at_min = fp_plus_fuel * hrAtMin + VC;
    
    // ramping costs
    bool generation_smaller_maxcap = state_Generation[i] < maxCap;
    // float ramp_up_to_max_cost = generation_smaller_maxcap * (rampUpSPin + rampUpCost / (maxCap * rampUpHorizon));
    float ramp_up_to_max_cost = generation_smaller_maxcap * (rampUpSPin + fdividef(rampUpCost, maxCap * rampUpHorizon));
    bool generation_larger_mindisp = state_Generation[i] > minDisp;
    // float ramp_down_to_min_cost = generation_larger_mindisp * (rampDownSPin + rampDownCost / (minDisp * rampDownHorizon));
    float ramp_down_to_min_cost = generation_larger_mindisp * (rampDownSPin + fdividef(rampDownCost, minDisp * rampDownHorizon));
    bool run_at_min_index = maxCap * (pp[i] - optimal_marginal_cost_at_max - ramp_up_to_max_cost) <
        minDisp * (pp[i] - optimal_marginal_cost_at_min - ramp_down_to_min_cost);
    bool not_run_at_min_index = !run_at_min_index;

    // compute total startup costs
    bool is_cold_start = state_hoursShut[i] >= coldStartup;
    bool is_not_cold_start = !is_cold_start;
    //  & used to be * 
    float fixed_and_fuel_startup_cost = is_cold_start * (fixedStartupCostCold + startFuelCold * fp[i]) + 
        is_not_cold_start * (fixedStartupCost + startFuel * fp[i]);

    // float startup_SP = startupSPin[i] + fixed_and_fuel_startup_cost / (startupHorizon * maxCap);
    float startup_SP = startupSPin[i] + fdividef(fixed_and_fuel_startup_cost, startupHorizon * maxCap);
    bool startup_profit_v = pp[i] - optimal_marginal_cost_at_max - startup_SP > 0.;
    bool do_startup = dc_canStart[i] * ((dc_forceStart[i] == 2) |
                                    (startup_profit_v & (dc_forceStart[i] == 1)));

    // compute shutdown
    float actual_gen_profit = run_at_min_index * (pp[i] - optimal_marginal_cost_at_min) * minDisp +
        not_run_at_min_index * (pp[i] - optimal_marginal_cost_at_max) * maxCap;
    float shutdown_gen_profit = shutdownHorizon * actual_gen_profit;
    float shut_cost_sp = shutdownHorizon * shutdownSPin * maxCap;
    bool is_shutdown_profitable = shutdown_gen_profit < - (fixed_and_fuel_startup_cost + shut_cost_sp);
    bool do_shutdown = dc_canShut[i] & ((dc_forceShut[i] == 2) |
                                    (is_shutdown_profitable & (dc_forceShut[i] == 1)));
    // compute dispatch
    bool not_state_state = !state_state[i];
    bool curr_state_tmp = (state_state[i] & (!do_shutdown)) | (not_state_state & do_startup);
    curr_state[i] = curr_state_tmp;
    bool state_change = (curr_state_tmp != state_state[i]);

    // accounting
    float curr_generation_tmp = curr_state_tmp * (maxCap * not_run_at_min_index + run_at_min_index * minDisp);
    float generation_change = curr_generation_tmp - state_Generation[i];
    // float ramping_adjustment = (0.5 / rampRate / hours_in_block) * fabs(generation_change) * generation_change;
    float ramping_adjustment = fdividef(0.5, rampRate * hours_in_block) * fabs(generation_change) * generation_change;
    float curr_generation = curr_generation_tmp - ramping_adjustment;
    float curr_energy = curr_generation * hours_in_block;
    float revenue = curr_energy * pp[i];
    float variable_cost = VC * curr_energy;
    float actual_heat_rate = run_at_min_index * hrAtMin + not_run_at_min_index * hrAtMax;
    float fuel_cost = curr_energy * fp_plus_fuel * actual_heat_rate;

    bool not_curr_state = !curr_state_tmp;
    bool starts_tmp = curr_state_tmp & not_state_state;  // curr_state > state_state[i]
    shuts_tmp = not_curr_state & state_state[i];

    float startup_cost = (is_cold_start & starts_tmp) * (fixedStartupCostCold + fp[i] * startFuelCold) +
                      (is_not_cold_start & starts_tmp) * (fixedStartupCost + fp[i] * startFuel);

    bool ramp_cost_up_ind = (!starts_tmp) & (generation_change > SMALL_EPS);
    bool ramp_cost_dn_ind = (!shuts_tmp) & (generation_change < - SMALL_EPS);
    float ramp_cost = ramp_cost_up_ind * rampUpCost + ramp_cost_dn_ind * rampDownCost;
    // cashflow 
    cashflow[i] = revenue - (fuel_cost + variable_cost + startup_cost + ramp_cost);

    // new unit state
    nus_hours_in_state[i] = state_hoursInState[i] * (!state_change) + hours_in_block;
    nus_generation[i] = curr_generation_tmp;
    nus_total_starts[i] = state_TotalStarts[i] + starts_tmp;
    nus_hours_shut[i] = not_curr_state * (state_hoursShut[i] + hours_in_block);
    nus_hours_run[i] = curr_state_tmp * (state_hoursRun[i] + hours_in_block);
    nus_global_starts[i] = state_globalStarts[i] + starts_tmp;
    nus_state[i] = curr_state_tmp;

"""


opd_k = ElementwiseKernel(inputs, loop_code, 
                          loop_prep=loop_prep,
                          name="opd_k",
                          preamble="#define SMALL_EPS 1e-5")
