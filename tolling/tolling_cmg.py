# simulation tolling model
import config
import numpy as np
import mrds
import ds
import opd_1fuel
if config.CUDA_PRESENT: 
    import pycuda.gpuarray as gpa
    import cuda_ops
    import opd_1fuel_cu


class tolling_params():
    pass


def tolling_market_tuple(tm_dict):
    days_block = tm_dict['days_block']
    hours_block = tm_dict['hours_block']
    hours_block_names = tm_dict['hours_block_names']
    days_block_names = tm_dict['days_block_names']
    power_bl_names = tm_dict['power_bl_names']
    fuel_idx_name = tm_dict['fuel_idx_name']
    cash_vols = tm_dict['cash_vols']

    return days_block, hours_block, hours_block_names, days_block_names, \
        power_bl_names, fuel_idx_name, cash_vols


def construct_consequitive_hours_fct(days_partition, hours_partition, nb_days,
                                     block_names_partition):
    """
    for the spot process
    :param days_partition  partition list [[0,1,2,3,4],[5,6]]
    :param hours_parition: hours list [[8, 16], [12,12]]
    :param block_names_partition: same as days_partition just for blocks
    """
    def belongs_to_group(day_week):
        idx_nb = 0
        for k in days_partition:
            if day_week in k:
                return idx_nb
            else:
                idx_nb += 1

    blocks_seq = []
    blocks_name_seq = []
    for day in range(nb_days):
        day_week = np.mod(day, 7)
        block_group_idx = belongs_to_group(day_week)
        blocks_seq += hours_partition[block_group_idx]
        blocks_name_seq += block_names_partition[block_group_idx]

    return np.array(blocks_seq, dtype=np.int32), blocks_name_seq


def tolling_power_fuel_process(fwd_date,
                               toll_start, toll_end,
                               nb_sim,
                               days_partition,
                               power_blocks_names,
                               hours_partition,
                               fuel_idx_name,
                               cash_vols,
                               nb_days,
                               debug_ind=False,
                               fixed_monthly=None,
                               cash_vols_overwrite=False,
                               parallel=False,
                               model_ind='skew',
                               adj_fwd_tenors_days=None,
                               adj_vol_tenors_days=None,
                               cash_fwd_tenors_days=None,
                               cash_vol_tenors_days=None,
                               manual_adj=None,
                               cash_corr_adj=None,
                               cuda_ind=False,
                               mm_overwrite=None, 
                               new_corr_mtx=None):

    """
    simulate spot prices in the tolling model.
    Inputs:
      fwd_date ... date for which to simulate '20141114'
      toll_start, toll_end ... dates in string for when the toll is given '20141114'
      nb_fwds ... number of fwd contracts to simulate (e.g. 10)
      nb_sim ... nb. sims
      days_partition ... partition of the week [[0,1,2,3,4],[5,6]]
      days_parition_names ... partition names ['weekday', 'weekend']
      power_block_names ... power blocks
        [['ERCOT_NORTH-PEAK', 'ERCOT_NORTH-OFFPEAK'],['ERCOT_NORTH-OFFPEAK', 'ERCOT_NORTH-OFFPEAK']]
      hours_partition ... inf orm [[6,18],[12,12]]
      hours_parition_names ... in form [['peak', 'offpeak'],['offpeak', 'offpeak']]
      debug_ind ... whether to debug

      INSERT MORE
    """
    # obtaining the months for calibration
    power_1 = power_blocks_names[0][0]
    fwd_tenors_dt = ds.get_forward_curve(power_1, fwd_date)[0]
    toll_end_dt = ds.convert_str_datetime(toll_end)
    toll_end_month = np.sum([ft < toll_end_dt for ft in fwd_tenors_dt])
    nb_fwds = toll_end_month

    power_gas_blocks = set([item
                            for sublist in power_blocks_names
                                for item in sublist])  # different blocks
    power_gas_blocks.add(fuel_idx_name)
    # mapping from names to number, works as: power_gas_blocks['ATSI_7X8'] gives 1 e.g.
    power_gas_block_idx = {pg_name: pg_idx for pg_name, pg_idx in
                           zip(power_gas_blocks, range(len(power_gas_blocks)))}
    if mm_overwrite is None:
        power_models = mrds.mrds_calib_multiple(list(power_gas_blocks),
                                                fwd_date,
                                                [nb_fwds+1] * len(power_gas_blocks),
                                                model_ind=model_ind,
                                                adj_fwd_tenors_days=adj_fwd_tenors_days,
                                                adj_vol_tenors_days=adj_vol_tenors_days)
    else:
        power_models = mm_overwrite

    if new_corr_mtx is not None:
        power_models.complete_corr_mat = new_corr_mtx

    power_gas_cash_vol_names = set([item
                                    for sublist in power_blocks_names
                                    for item in sublist])  # different blocks
    power_gas_cash_vol_names.add(fuel_idx_name)
    fwd_date_dt = ds.convert_str_datetime(fwd_date)

    for days_bl, days_cv, fuel_cv in zip(power_blocks_names,
                                         cash_vols['power'],
                                         cash_vols['fuel']):
        for hour_bl, hour_cv, hour_fuel_cv in zip(days_bl, days_cv, fuel_cv):
            pg_idx = power_gas_block_idx[hour_bl]
            if not cash_vols_overwrite:
                adj_fwd_tenors, adj_vol_tenors = mrds.find_adj_tenors(pg_idx, cash_fwd_tenors_days,
                                                                      cash_vol_tenors_days)
                fwd_vol_tenors_numeric, fwd_vol_values_unexpired, fwd_vol_tenors_code, \
                    fwd_vol_tenors = ds.get_fwd_vol_curve_numeric_tenor(hour_cv, fwd_date, fwd_date_dt,
                                                                        fwd_vol_ind='vol',
                                                                        adj_fwd_tenors_days=adj_fwd_tenors,
                                                                        adj_vol_tenors_days=adj_vol_tenors)
                cash_vol_write = np.array(fwd_vol_values_unexpired[:(nb_fwds+1)],
                                          dtype=np.double)
                power_models.set_cash_vols(pg_idx, cash_vol_write)
            else:
                power_models.set_cash_vols(pg_idx, cash_vols)

    fuel_com_nb = power_models.nb_assets - 1
    adj_fwd_tenors, adj_vol_tenors = mrds.find_adj_tenors(fuel_com_nb, cash_fwd_tenors_days,
                                                          cash_vol_tenors_days)
    fwd_vol_tenors_numeric, fwd_vol_values_unexpired, fwd_vol_tenors_code, \
        fwd_vol_tenors = ds.get_fwd_vol_curve_numeric_tenor(cash_vols['fuel'][0][0], fwd_date, fwd_date_dt,
                                                            fwd_vol_ind='vol',
                                                            adj_fwd_tenors_days=adj_fwd_tenors,
                                                            adj_vol_tenors_days=adj_vol_tenors)
    cash_vols_fuel = np.array(fwd_vol_values_unexpired[:(nb_fwds+1)], dtype=np.double)
    power_models.set_cash_vols(power_gas_block_idx[fuel_idx_name], cash_vols_fuel)
    if manual_adj is not None:
        exec(manual_adj)
    return tolling_power_fuel_process_reduced(toll_start, toll_end,
                                              nb_sim,
                                              days_partition,
                                              hours_partition,
                                              fuel_idx_name,
                                              power_blocks_names,
                                              power_models,
                                              power_gas_block_idx,
                                              nb_days,
                                              debug_ind=debug_ind,
                                              fixed_monthly=fixed_monthly,
                                              parallel=parallel,
                                              cuda_ind=cuda_ind), power_gas_block_idx


def tolling_power_fuel_process_reduced(toll_start, toll_end,
                                       nb_sim,
                                       days_partition, hours_partition,
                                       fuel_idx_name,
                                       power_bl_names, power_models,
                                       power_gas_block_idx,
                                       nb_days,
                                       debug_ind=False,
                                       fixed_monthly=None,
                                       parallel=False,
                                       cuda_ind=False):

    fwd_tenors_dt = power_models.forward_tenors_dt_list[0]
    toll_start_dt = ds.convert_str_datetime(toll_start)
    toll_end_dt = ds.convert_str_datetime(toll_end)
    toll_start_month = max(np.sum([ft < toll_start_dt for ft in fwd_tenors_dt])-1, 0)
    toll_end_month = max(np.sum([ft < toll_end_dt for ft in fwd_tenors_dt])-1, 0)
    tenors_chosen = range(toll_start_month, toll_end_month+1)

    fom_sims_all = [power_models.simulate_curves_fom(asset_nb, nb_sim,
                                                     tenors_list=tenors_chosen,
                                                     cuda_ind=cuda_ind)
                    for asset_nb in range(power_models.nb_assets)]
    power_fuel_foms = [[(fom_sims_all[power_gas_block_idx[mo]],
                         fom_sims_all[power_gas_block_idx[fuel_idx_name]])
                        for mo in model_block_l]
                       for model_block_l in power_bl_names]

    return {'tenors_chosen': tenors_chosen,
            'power_models': power_models,
            'fom_sims_all': fom_sims_all,
            'power_fuel_foms': power_fuel_foms}


class tolling_model_CMG():
    def __init__(self,
                 fwd_date,
                 toll_start, toll_end,
                 nb_sim,
                 power_blocks_names, fuel_idx_name,
                 days_partition, days_partition_names,
                 hours_partition, hours_partition_names,
                 cash_vols,
                 params,
                 debug_ind=False,
                 cash_vols_overwrite=False,
                 revenue_put=False,
                 power_spot_model_given=None,
                 model_ind='skew',
                 adj_fwd_tenors_days=None,
                 adj_vol_tenors_days=None,
                 cash_fwd_tenors_days=None,
                 cash_vol_tenors_days=None,
                 manual_adj=None,
                 cash_corr_adj=None,
                 cuda_ind=False,
                 mm_overwrite=None):
        self.params = params
        self.debug_ind = debug_ind
        self.revenue_put = revenue_put
        self.nb_sim = nb_sim
        self.nb_days = params.nb_days
        self.fwd_date = fwd_date
        self.toll_start = toll_start
        self.toll_end = toll_end
        self.power_blocks_names = power_blocks_names
        self.adj_fwd_tenors_days = adj_fwd_tenors_days
        self.adj_vol_tenors_days = adj_vol_tenors_days
        self.cash_fwd_tenors_days = cash_fwd_tenors_days
        self.cash_vol_tenors_days = cash_vol_tenors_days
        self.manual_adj = manual_adj
        self.cash_corr_adj = cash_corr_adj
        self.cuda_ind = cuda_ind

        power_gas_blocks = set([item
                                for sublist in power_blocks_names
                               for item in sublist])
        power_gas_blocks.add(fuel_idx_name)
        self.power_gas_block_idx = {pg_name: pg_idx for pg_name, pg_idx in
                                    zip(power_gas_blocks, range(len(power_gas_blocks)))}

        self.fuel_idx_name = fuel_idx_name
        self.days_partition = days_partition
        self.days_partition_names = days_partition_names
        self.hours_partition = hours_partition
        self.hours_partition_names = hours_partition_names
        self.cash_vols = cash_vols
        self.cash_vols_overwrite = cash_vols_overwrite
        if self.fuel_idx_name is not 'FIXED':
            fixed_monthly_val = None
        else:
            fixed_monthly_val = self.params.fixedCostPerMonth

        if power_spot_model_given is None:
            spots_models, power_gas_block_idx = \
                tolling_power_fuel_process(fwd_date,
                                           self.toll_start, self.toll_end,
                                           nb_sim,
                                           self.days_partition,
                                           self.power_blocks_names,
                                           self.hours_partition,
                                           self.fuel_idx_name,
                                           self.cash_vols,
                                           self.nb_days,
                                           debug_ind=self.debug_ind,
                                           fixed_monthly=fixed_monthly_val,
                                           cash_vols_overwrite=self.cash_vols_overwrite,
                                           parallel=True,
                                           model_ind=model_ind,
                                           adj_fwd_tenors_days=self.adj_fwd_tenors_days,
                                           adj_vol_tenors_days=self.adj_vol_tenors_days,
                                           cash_fwd_tenors_days=self.cash_fwd_tenors_days,
                                           cash_vol_tenors_days=self.cash_vol_tenors_days,
                                           manual_adj=self.manual_adj,
                                           cash_corr_adj=self.cash_corr_adj,
                                           cuda_ind=self.cuda_ind,
                                           mm_overwrite=mm_overwrite)
        else:
            power_models, ng_model = power_spot_model_given
            spots_models = tolling_power_fuel_process_reduced(self.toll_start, self.toll_end,
                                                              nb_sim,
                                                              self.days_partition,
                                                              self.hours_partition,
                                                              self.fuel_idx_name,
                                                              self.power_blocks_names,
                                                              power_models,
                                                              self.power_gas_block_idx,
                                                              self.nb_days,
                                                              debug_ind=self.debug_ind,
                                                              fixed_monthly=fixed_monthly_val,
                                                              parallel=False,
                                                              cuda_ind=self.cuda_ind)
        self.power_models = spots_models['power_models']
        self.fom_sims_all = spots_models['fom_sims_all']
        self.power_fuel_foms = spots_models['power_fuel_foms']
        self.tenors_chosen = spots_models['tenors_chosen']

        # tolling support vectors
        self.days_toll, self.days_d_toll, self.days_diff_toll, self.days_diff_l_toll = \
            self.power_models.generate_days_vecs(self.hours_partition, self.days_partition,
                                                 cuda_ind=self.cuda_ind)

        params_prelim = [self.params.hrAtMax,
                         self.params.hrAtMin,
                         self.params.maxCap,
                         self.params.minDisp,
                         self.params.startFuel,
                         self.params.startFuelCold,
                         self.params.addFuelCost,
                         self.params.VC,
                         self.params.rampRate,
                         self.params.shutdownSPin,
                         self.params.minDownTime,
                         self.params.minRunTime,
                         self.params.fixedStartupCost,
                         self.params.fixedStartupCostCold,
                         self.params.maxMonthlyStarts,
                         self.params.coldStartup,
                         self.params.startupHorizon,
                         self.params.shutdownHorizon,
                         self.params.rampUpSPin,
                         self.params.rampDownSPin,
                         self.params.rampUpCost,
                         self.params.rampDownCost,
                         self.params.rampUpHorizon,
                         self.params.rampDownHorizon]
        
        if self.cuda_ind:
            self.params_used = gpa.to_gpu(np.array(params_prelim)).astype(np.float32)
        else:
            self.params_used = params_prelim

    def construct_consequitive_hours(self):
        self.blocks_seq, self.block_names_seq = construct_consequitive_hours_fct(self.days_partition,
                                                                                 self.hours_partition,
                                                                                 self.nb_days,
                                                                                 self.hours_partition_names)
        return self.blocks_seq, self.block_names_seq

    def generate_spots(self, m):
        """
        generates spots from month m
        """
        days_tuple = (self.days_toll, self.days_d_toll, self.days_diff_toll, self.days_diff_l_toll)
        spot_blocks_m = [self.power_models.simulate_spot_blocks_from_fom(self.fom_sims_all,
                                                                         asset_nb,
                                                                         m, self.nb_sim,
                                                                         self.days_partition,
                                                                         self.hours_partition,
                                                                         days_tuple,
                                                                         tenors_chosen=self.tenors_chosen,
                                                                         cuda_ind=self.cuda_ind)
                         for asset_nb in range(self.power_models.nb_assets)]
        # power fuel spots for month m
        pf_spots_m = [[(spot_blocks_m[self.power_gas_block_idx[mo]],
                        spot_blocks_m[self.power_gas_block_idx[self.fuel_idx_name]])
                       for mo in model_block_l]
                      for model_block_l in self.power_blocks_names]

        total_nb_blocks = 0
        for day in range(self.nb_days):
            day_week = np.mod(day, 7)
            for dp, psim in zip(self.days_partition, pf_spots_m):
                if day_week in dp:
                    total_nb_blocks += len(psim)

        if not self.cuda_ind:
            sim_m = np.empty((total_nb_blocks, self.nb_sim))
            sim_m_fuel = np.empty((total_nb_blocks, self.nb_sim))
        else:
            sim_m = gpa.empty((total_nb_blocks, self.nb_sim), dtype=np.float32)
            sim_m_fuel = gpa.empty((total_nb_blocks, self.nb_sim), dtype=np.float32)
        block_count = 0
        for day in range(self.nb_days):
            day_week = np.mod(day, 7)
            for dp, psim in zip(self.days_partition, pf_spots_m):
                if day_week in dp:
                    for ms, fs in psim:
                        sim_m[block_count, :] = ms[day, :]
                        if self.fuel_idx_name != 'FIXED':
                            sim_m_fuel[block_count, :] = fs[day, :]
                        else:
                            # WRONG WRONG self.power_models.fixed_monthly[m]  # THIS DOESNT WORK ON CUDA
                            sim_m_fuel[block_count, :] = 1.
                        block_count += 1
        power_sim = sim_m
        fuel_sim = sim_m_fuel

        return power_sim, fuel_sim

    def startup_decision(self, cs, params, dispatch_mode='cmg', ci=False):
        """
        routine deciding whether startup is done
        """
        if dispatch_mode == 'cmg':
            if not ci:
                cnd_1 = cs.total_starts < params.maxMonthlyStarts
                cnd_2 = cs.hours_shut >= params.minDownTime
                cs.can_start = cnd_1 & cnd_2
            else:  # this kernel implements exactly what is above 3 lines 
                cs.can_start = cuda_ops.comp_two_arrays_and(cs.total_starts, cs.hours_shut,
                                                            params.maxMonthlyStarts,
                                                            params.minDownTime)
        elif dispatch_mode == 'peak_only':
            cnd_2 = cs.block_name == 'peak'
            cnd_1 = cs.total_starts < params.maxMonthlyStarts
            cnd_3 = cs.hours_shut >= params.minDownTime
            if not ci:
                return cnd_1 & cnd_2 & cnd_3
            else:
                return cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)
        elif dispatch_mode == 'offpeak_only':
            cnd_2 = cs.block_name != 'peak'
            cnd_1 = cs.total_starts < params.maxMonthlyStarts
            cnd_3 = cs.hours_shut >= params.minDownTime
            if not ci:
                return cnd_1 & cnd_2 & cnd_3
            else:
                return cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def forced_startup(self, cs, pp, fp, dv=None, dispatch_mode='cmg', ci=False):
        """
        try to force power plant to start
        :param cs: current state object
        :param dispatch_mode: which dispatch to follow
        :param ci: cuda indicator (True or False)
        """
        if dispatch_mode == 'mrg':
            decision_1 = pp - self.params.hrAtMax * fp
            cs.force_start = 2 * (decision_1 > dv[1]) + (decision_1 > dv[0]) & (decision_1 < dv[1])
        elif dispatch_mode == 'peak_only':
            cs.force_start.fill(2 * (cs.block_name == 'peak'))
        elif dispatch_mode == 'offpeak_only':
            cs.force_start.fill(2 * (cs.block_name != 'peak'))

    def shutdown_decision(self, cs, params, dispatch_mode='cmg', ci=False):
        """
        decision whether it is sensible to shut down
        :param cs: current state, cs.can_shut is filled by this routine
           cs.can_shut is array of bools
        """
        if dispatch_mode == 'cmg':
            if not ci:
                cs.can_shut = cs.hours_run >= params.minRunTime
            else:
                cs.can_shut = cuda_ops.comp_array_number(cs.hours_run, params.minRunTime,
                                                         op='larger', dtype='int32')
        elif dispatch_mode == 'peak_only':
            cnd_2 = cs.block_name != 'peak'
            cnd_3 = cs.hours_run >= params.minRunTime
            if not ci:
                cs.can_shut = cnd_2 & cnd_3
            else:
                cnd_3_int = cnd_3.astype(np.int32)
                cuda_ops.min_int_two(cnd_2, cnd_3_int, cs.can_shut)
        elif dispatch_mode == 'offpeak_only':
            cnd_2 = cs.block_name == 'peak'
            cnd_3 = cs.hours_run >= params.minRunTime
            if not ci:
                cs.can_shut = cnd_2 & cnd_3
            else:
                cuda_ops.min_int_two(cnd_2, cnd_3, cs.can_shut)

    def forced_shutdown(self, cs, pp, fp, dv=None, dispatch_mode='cmg', ci=False):
        """
        decision to forcefully shut down, can take 3 outcomes: 2, 1, 0
        """
        if dispatch_mode == 'mrg':
            decision_1 = pp - self.params.hrAtMax * fp
            cs.force_shut = 2 * (decision_1 < dv[3]) + (decision_1 > dv[2]) & (decision_1 < dv[3])
        elif dispatch_mode == 'peak_only':
            cs.force_shut.fill(2 * (cs.block_name != 'peak'))
        elif dispatch_mode == 'offpeak_only':
            cs.force_shut.fill(2 * (cs.block_name != 'offpeak'))

    def set_other_params(self, cs, pl, dispatch_mode='cmg', ci=False):
        """
        set up the cs.force_ and cs.can_ parameters
        """
        if dispatch_mode == 'cmg' or dispatch_mode == 'mrg':
            if not ci:
                cs.force_start = np.ones(pl, dtype=np.short)
                cs.force_shut = np.ones(pl, dtype=np.short)
                cs.can_start = np.empty(pl, dtype=np.short)  # CHECK THIS ONE
                cs.can_shut = np.empty(pl, dtype=np.short)
            else:  # cuda
                # cs.force_start = gpa.zeros(pl, dtype=np.int32) + 1  # (bottom two lines implement this)
                cs.force_start = gpa.empty(pl, dtype=np.int32)
                cs.force_start.fill(1)
                # cs.force_shut = gpa.zeros(pl, dtype=np.int32) + 1
                cs.force_shut = gpa.empty(pl, dtype=np.int32)
                cs.force_shut.fill(1)
                cs.can_start = gpa.empty(pl, dtype=bool)
                cs.can_shut = gpa.empty(pl, dtype=bool)
        elif dispatch_mode == 'always_run':
            if not ci:  # cpu
                cs.force_start = np.empty(pl, dtype=np.short)
                cs.force_start.fill(2)
                cs.force_shut = np.zeros(pl, dtype=np.short)
                cs.can_start = np.empty(pl, dtype=np.short)
                cs.can_start.fill(1)
                cs.can_shut = np.empty(pl, dtype=np.short)
                cs.can_shut.fill(0)
            else:  # cuda
                cs.force_shut = gpa.empty(dtype=np.int32)  # force shut done once
                cs.force_shut.fill(0)
                cs.force_start = gpa.empty(dtype=np.int32)  # force start set here
                cs.force_start.fill(2)
                cs.can_shut = gpa.zeros(pl, dtype=bool)
                cs.can_start = gpa.zeros(pl, dtype=bool) + 1
        else:  # peak & offpeak only
            if not ci:  # cpu
                cs.force_start = np.empty(pl, dtype=np.short)
                cs.force_shut = np.empty(pl, dtype=np.short)
                cs.can_start = np.empty(pl, dtype=np.short)
                cs.can_shut = np.empty(pl, dtype=np.short)
            else:  # cuda
                cs.force_start = 1  # TO FIX FIX FIX FIX
                cs.force_shut = 1
                cs.can_start = 1
                cs.can_shut = 1

    def dispatch_month(self, m, conseq_hours, conseq_block_names,
                       pl, power_spots, fuel_spots, dv=None,
                       dispatch_mode='cmg'):
        """
        :param dv: decision variable, for optimization
        """
        cs = tolling_params()
        if not self.cuda_ind:  # CPU
            nus_hours_in_state = np.zeros(pl, dtype=np.int16)
            nus_generation = np.zeros(pl)
            nus_total_starts = np.zeros(pl, dtype=np.short)
            nus_hours_shut = np.empty(pl, dtype=np.short)
            nus_hours_shut.fill(self.params.MDT)
            nus_hours_run = np.empty(pl, dtype=np.short)
            nus_hours_run.fill(self.params.MUT)
            nus_global_starts = np.zeros(pl, dtype=np.short)
            nus_state = np.zeros(pl, dtype=np.short)
            cf_per_path_tmp = np.empty(pl)  # cash flow per path
            cf_per_path = np.zeros(pl)  # cumulative cash flows
            curr_state = np.empty(pl, dtype=bool)
        else:  # GPU
            nus_hours_in_state = gpa.zeros(pl, dtype=np.int32)
            nus_generation = gpa.zeros(pl, dtype=np.float32)
            nus_total_starts = gpa.zeros(pl, dtype=np.int32)
            nus_hours_shut = gpa.empty(pl, dtype=np.int32)
            nus_hours_shut.fill(self.params.MDT)
            nus_hours_run = gpa.zeros(pl, dtype=np.int32)
            nus_hours_run.fill(self.params.MUT)
            nus_global_starts = gpa.zeros(pl, dtype=np.int32)
            nus_state = gpa.zeros(pl, dtype=bool)  # initial state, not running
            # cf_per_path_tmp = gpa.empty(9*pl, dtype=np.float32)
            cf_per_path_tmp = gpa.empty(pl, dtype=np.float32)
            cf_per_path = gpa.zeros(pl, dtype=np.float32)
            curr_state = gpa.empty(pl, dtype=bool)

        cs.dispatch_mode = dispatch_mode  # cmg, mrg, or always run
        self.set_other_params(cs, pl, dispatch_mode=dispatch_mode, ci=self.cuda_ind)
        cs.df = 1.

        for spot_idx, (block_hours, block_name) in enumerate(zip(conseq_hours, conseq_block_names)):
            power_prices, fuel_prices = power_spots[spot_idx, :], fuel_spots[spot_idx, :]
            # new states are previous states
            if not self.cuda_ind:
                cs.hours_block = np.double(block_hours)
            else:
                cs.hours_block = np.int32(block_hours)

            cs.hours_in_state = nus_hours_in_state
            cs.state = nus_state
            cs.generation = nus_generation
            cs.total_starts = nus_total_starts
            cs.hours_shut = nus_hours_shut
            cs.hours_run = nus_hours_run
            cs.global_starts = nus_global_starts
            cs.block_name = block_name

            if dispatch_mode == 'cmg':
                start_cost_init = self.params.fixedStartupCostCold + \
                                  self.params.SF * (fuel_prices + self.params.addFuelCost) - \
                                  self.params.E_S * power_prices
                # shadow prices
                startup_sp_in = start_cost_init / (self.params.maxCap * self.params.startupHorizon)
            else:
                startup_sp_in = 0.
            self.params.startupSPin = startup_sp_in

            # the reason why first are overwritten is that they change very often
            self.startup_decision(cs, self.params, dispatch_mode=dispatch_mode,
                                  ci=self.cuda_ind)
            self.shutdown_decision(cs, self.params, dispatch_mode=dispatch_mode,
                                   ci=self.cuda_ind)
            # the following updates cs.force_shut, cs.force_start
            self.forced_startup(cs, power_prices, fuel_prices, dv, dispatch_mode=dispatch_mode,
                                ci=self.cuda_ind)
            self.forced_shutdown(cs, power_prices, fuel_prices, dv, dispatch_mode=dispatch_mode,
                                 ci=self.cuda_ind)
            nb_paths = len(power_prices)
            if not self.cuda_ind:
                opd_f = opd_1fuel.opd_1fuel
            else:
                opd_f = opd_1fuel_cu.opd_k

            opd_f(spot_idx,
                  power_prices, fuel_prices,
                  self.params_used,
                  startup_sp_in,
                  cs.state,
                  cs.hours_in_state,
                  cs.generation,
                  cs.total_starts,
                  cs.hours_shut,
                  cs.hours_run,
                  cs.global_starts,
                  cs.can_start,
                  cs.can_shut,
                  cs.force_start,
                  cs.force_shut,
                  cs.hours_block,
                  cs.df,
                  nb_paths,
                  nus_hours_in_state,
                  nus_generation,
                  nus_total_starts,
                  nus_hours_shut,
                  nus_hours_run,
                  nus_global_starts,
                  nus_state,
                  cf_per_path_tmp,
                  curr_state)

            cf_per_path += cf_per_path_tmp
                
        df_m = self.power_models._discount_discount[m]
        dispatch_results_df = df_m * cf_per_path
        return dispatch_results_df

    def dispatch_cmg(self, dispatch_mode='cmg'):
        """
        dispatch algorithm
        """
        months_to_compute = self.tenors_chosen
        conseq_hours, conseq_block_names = construct_consequitive_hours_fct(self.days_partition,
                                                                            self.hours_partition,
                                                                            self.nb_days,
                                                                            self.hours_partition_names)
        if not self.cuda_ind:
            dispatch_results_total = np.zeros(self.nb_sim)
        else:
            dispatch_results_total = gpa.zeros(self.nb_sim, dtype=np.float32)

        dispatch_res = {}
        for m in months_to_compute:
            ps, fs = self.generate_spots(m)
            dispatch_res[m] = self.dispatch_month(m, conseq_hours, conseq_block_names,
                                                  self.nb_sim,
                                                  ps, fs,
                                                  dispatch_mode=dispatch_mode)
        for m in months_to_compute:
            dispatch_results_total += dispatch_res[m]

        if self.revenue_put:
            cf_total_per_path_positive = dispatch_results_total
            revenue_put_per_path = np.maximum(self.params.revenue_strike - cf_total_per_path_positive, 0.)
            revenue_put_total = np.mean(revenue_put_per_path)

        # self.computed_basic_toll = True
        if not self.cuda_ind:
            self.dispatch_res_months = {m: np.mean(dispatch_res[m]) for m in months_to_compute}
        else:
            self.dispatch_res_months = {m: gpa.sum(dispatch_res[m])/dispatch_res[m].size 
                                        for m in months_to_compute}

        return_dict = {'cashflow': sum(self.dispatch_res_months.values())}
        if self.revenue_put:
            return_dict['revenue_put_total'] = revenue_put_total
            return return_dict
        else:
            return return_dict

    def resimulate_prices(self):
        resimulating = tolling_power_fuel_process_reduced(self.toll_start, self.toll_end,
                                                          self.nb_sim,
                                                          self.days_partition,
                                                          self.hours_partition,
                                                          self.fuel_idx_name,
                                                          self.power_blocks_names,
                                                          self.power_models,
                                                          self.power_gas_block_idx,
                                                          self.nb_days)
        self.power_spots = resimulating['power_spot']
        self.fuel_spots = resimulating['fuel_spot']
