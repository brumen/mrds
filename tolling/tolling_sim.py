import config
import numpy as np
import tolling


class tolling_sim():
    """
    simulation tolling model
    """
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
                 cash_corr_adj=None):
        """
        toll_start, toll_end ... months
        power_blocks_names ... list of list [ ['ATSI', 'asd'], ['asd', 'asd']]
        hours_partition ...
        """
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
        self.blocks_seq = tolling.construct_consequitive_hours_fct(self.days_partition,
                                                                   self.hours_partition,
                                                                   self.nb_days)

        if self.fuel_idx_name is not 'FIXED':
            fixed_monthly_val = None
        else:
            fixed_monthly_val = self.params.fixedCostPerMonth

        if power_spot_model_given is None:
            spots_models, power_gas_block_idx = tolling.tolling_power_fuel_process(fwd_date,
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
                                                    cash_corr_adj=self.cash_corr_adj)
        else:
            power_models, ng_model = power_spot_model_given
            spots_models = tolling.tolling_power_fuel_process_reduced(self.toll_start, self.toll_end,
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
                                                                      parallel=False)

        self.power_models = spots_models['power_models']
        self.power_spots = spots_models['power_spot']
        self.fuel_spots = spots_models['fuel_spot']

    def dispatch_month(self, m, conseq_hours, pl):
        return self.dispatch_month_raw(m, conseq_hours, pl, self.power_spots, self.fuel_spots)

    def dispatch_month_raw(self, m, conseq_hours, pl, power_spots, fuel_spots):

        def startup_decision(cs, params, always_run=False):
            nb_sim = len(cs.state)
            if always_run:
                return np.ones(nb_sim)
            else:
                cnd_1 = cs.total_starts < params.maxMonthlyStarts
                cnd_2 = cs.hours_shut >= params.minDownTime
                return cnd_1 & cnd_2

        def shutdown_decision(cs, params, always_run=False):
            nb_sim = len(cs.state)
            if always_run:
                return np.ones(nb_sim)
            else:
                cnd_1 = cs.hours_run >= params.minRunTime
                return cnd_1

        NewUnitState = np.empty((pl, 7))
        NewUnitState[:, 0] = np.zeros(pl)  # hours in state
        NewUnitState[:, 1] = np.ones(pl)  #initial state - running
        NewUnitState[:, 2] = np.zeros(pl)  # generation
        NewUnitState[:, 3] = np.zeros(pl)  # total starts
        NewUnitState[:, 4] = np.ones(pl) * self.params.MDT  # hours shut - so that it can start right away
        NewUnitState[:, 5] = np.ones(pl) * self.params.MUT  # hours run
        NewUnitState[:, 6] = np.zeros(pl)  # hours in state

        CashFlowPerPath_tmp = np.zeros(pl)
        CashFlowPerPath = np.zeros(pl)
        DispatchResults_tmp = np.zeros(12)
        cs = tolling.tolling_params()
        nb_blocks = len(conseq_hours)
        avg_cf_per_block = np.empty(nb_blocks)

        dispatch_results = 0.
        for spot_idx, block_hours in enumerate(conseq_hours):
            power_prices = power_spots[m][:, spot_idx]
            fuel_prices = fuel_spots[m][:, spot_idx]

            cs.hours_in_state = NewUnitState[:, 0]
            cs.state = NewUnitState[:, 1]
            cs.generation = NewUnitState[:, 2]
            cs.total_starts = NewUnitState[:, 3]
            cs.hours_shut = NewUnitState[:, 4]
            cs.hours_run = NewUnitState[:, 5]
            cs.global_starts = NewUnitState[:, 6]

            # startupCost
            StartCostInit = self.params.fixedStartupCostCold + \
                self.params.SF * (fuel_prices + self.params.addFuelCost) - \
                self.params.E_S * power_prices
            self.params.startupSPin = StartCostInit / self.params.maxCap / self.params.startupHorizon

            cs.can_start = startup_decision(cs, self.params)
            cs.can_shut = shutdown_decision(cs, self.params)
            cs.df = 1.
            cs.hours_block = np.double(block_hours)
            opd(power_prices, fuel_prices, cs, self.params,
                NewUnitState, DispatchResults_tmp, CashFlowPerPath_tmp)  # 1 block optimization
            CashFlowPerPath += CashFlowPerPath_tmp
            dispatch_results += DispatchResults_tmp
            avg_cf_per_block[spot_idx] = np.average(CashFlowPerPath_tmp)
            DF_m = self.power_models._discount_discount[m]
            DispatchResultsDF = DF_m * dispatch_results

            return {"DispatchResults": dispatch_results,
                    "DispatchResultsDF": DispatchResultsDF,
                    "CashFlow": CashFlowPerPath,
                    "avg_cf_per_block": avg_cf_per_block}

    def dispatch_fast(self, parallel=False):
        power_keys = self.power_models.keys()
        first_month_compute = power_keys[0]
        last_month_compute = power_keys[-1]
        nb_sim = self.power_spots[first_month_compute].shape[0]
        conseq_hours = tolling.construct_consequitive_hours_fct(self.days_partition,
                                                                self.hours_partition,
                                                                self.nb_days)
        DispatchResultsTotal = np.zeros(12)
        months_to_compute = range(first_month_compute, last_month_compute+1)
        self.months_to_compute = months_to_compute
        dispatch_res = [self.dispatch_month(m, conseq_hours, nb_sim)
                        for m in months_to_compute]




    def resimulate_prices(self):
        resimulating = tolling.tolling_power_fuel_process_reduced(self.toll_start, self.toll_end,
                                                                  self.nb_sim,
                                                                  self.days_partition,
                                                                  self.hours_partition
                                                                  NOT FINISHED
                                                                  )










