# simulation tolling model
import config

import datetime, numpy as np
import mrds, mrds_utils, ds, opd.opd_1fuel as opd_1fuel

if config.CUDA_PRESENT:
    import pycuda.gpuarray as gpa
    import cuda.cuda_ops as cuda_ops
    import opd.opd_1fuel_cu as opd_1fuel_cu


class TollingModel(object):

    def __init__(self
                 , calcDate   : datetime.date
                 , toll_start : datetime.date
                 , toll_end   : datetime.date
                 , nb_sim
                 , power_blocks_names
                 , fuel_idx_name
                 , days_partition
                 , days_partition_names
                 , hours_partition
                 , hours_partition_names
                 , cash_vols
                 , powerPlantParams
                 , debug_ind              = False
                 , cash_vols_overwrite    = False
                 , adj_fwd_tenors_days    = None
                 , adj_vol_tenors_days    = None
                 , cash_fwd_tenors_days   = None
                 , cash_vol_tenors_days   = None
                 , manual_adj             = None
                 , cash_corr_adj          = None):
        """
        Tolling model.

        :param calcDate: calculation date of the tolling model.
        :param toll_start: Start of the tolling deal.
        :param toll_end: End of the tolling deal
        :param powerPlantParams: parameters of the power plant used for dispatch. This should have the
                                 following fields:  they are of different types
                                   hrAtMax,
                                   hrAtMin,
                                   maxCap,
                                   minDisp,
                                   startFuel,
                                   startFuelCold,
                                   addFuelCost,
                                   VC,
                                   rampRate,
                                   shutdownSPin,
                                   minDownTime,
                                   minRunTime,
                                   fixedStartupCost,
                                   fixedStartupCostCold,
                                   maxMonthlyStarts,
                                   coldStartup,
                                   startupHorizon,
                                   shutdownHorizon,
                                   rampUpSPin,
                                   rampDownSPin,
                                   rampUpCost,
                                   rampDownCost,
                                   rampUpHorizon,
                                   rampDownHorizon


        """

        self.powerPlantParams      = powerPlantParams
        self.debug_ind   = debug_ind
        self.nb_sim      = nb_sim
        self.nb_days     = powerPlantParams.nb_days
        self.calcDate    = calcDate
        self.toll_start  = toll_start
        self.toll_end    = toll_end
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

        if self.fuel_idx_name is not 'FIXED':
            fixed_monthly_val = None
        else:
            fixed_monthly_val = self.powerPlantParams['fixedCostPerMonth']

        # tolling support vectors
        self.days_toll, self.days_d_toll, self.days_diff_toll, self.days_diff_l_toll = \
            self.powerModels.generate_days_vecs( self.hours_partition
                                               , self.days_partition
                                               , cuda_ind = self.cuda_ind)

        # for usage w/ this class
        self.__power_spot_model = None
        self.__powerModels      = None

    @property
    def power_spot_model(self):

        if self.__power_spot_model:
            return self.__power_spot_model

        return self.tolling_power_fuel_process(self.calcDate,
                                       self.toll_start, self.toll_end,
                                       nb_sim,
                                       self.days_partition,
                                       self.power_blocks_names,
                                       self.hours_partition,
                                       self.fuel_idx_name,
                                       self.cash_vols,
                                       self.nb_days,
                                       fixed_monthly       = fixed_monthly_val,
                                       cash_vols_overwrite = self.cash_vols_overwrite,
                                       parallel            = True,
                                       adj_fwd_tenors_days = self.adj_fwd_tenors_days,
                                       adj_vol_tenors_days = self.adj_vol_tenors_days,
                                       cash_fwd_tenors_days=self.cash_fwd_tenors_days,
                                       cash_vol_tenors_days=self.cash_vol_tenors_days,
                                       manual_adj=self.manual_adj,
                                       cash_corr_adj=self.cash_corr_adj,
                                       cuda_ind=self.cuda_ind )

    @power_spot_model.setter
    def power_spot_model(self, newPowerSpotModel):
        self.__power_spot_model = newPowerSpotModel

    @staticmethod
    def construct_consequitive_hours( days_partition
                                    , hours_partition
                                    , nb_days : int
                                    , block_names_partition ):
        """
        Construct consequitive hours for the tolling model, used for spot simulation process.

        :param days_partition:  partition list [[0,1,2,3,4],[5,6]]
        :type days_partition: list[list[int]]
        :param hours_parition: hours list [[8, 16], [12,12]]
        :type hours_partition: list[list[int]]
        :type nb_days: number of days
        :param block_names_partition: same as days_partition just for blocks
        :type block_names_partition: list[list[int]]
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

    def tolling_power_fuel_process( self
                                  , nb_sim
                                  , days_partition
                                  , power_blocks_names
                                  , hours_partition
                                  , fuel_idx_name
                                  , cash_vols
                                  , nb_days
                                  , fixed_monthly=None
                                  , cash_vols_overwrite=False
                                  , parallel=False
                                  , cash_fwd_tenors_days=None
                                  , cash_vol_tenors_days=None
                                  , manual_adj=None
                                  , cuda_ind = False ):

        """
        Simulate spot prices in the tolling model, by blocks.

        :param nb_sim: nb. simulations
        :param days_partition: partition of the week, e.g. [[0,1,2,3,4],[5,6]]
        :param days_parition_names: partition names ['weekday', 'weekend']
        :param power_block_names: power blocks
            as in [ ['ERCOT_NORTH-PEAK'   , 'ERCOT_NORTH-OFFPEAK']
                  , ['ERCOT_NORTH-OFFPEAK', 'ERCOT_NORTH-OFFPEAK']]
        :param hours_partition: in form [[6,18],[12,12]]
        :param hours_parition_names: in form [['peak', 'offpeak'],['offpeak', 'offpeak']]
        :param debug_ind: whether to debug

        """

        # obtaining the months for calibration
        toll_end_month = np.sum([ft < self.toll_end
                                 for ft in ds.get_forward_curve(power_blocks_names[0][0], self.calcDate)[0]])
        nb_fwds = toll_end_month

        power_gas_blocks = set([item
                                for sublist in power_blocks_names
                                for item in sublist])  # different blocks
        power_gas_blocks.add(fuel_idx_name)
        # mapping from names to number, works as: power_gas_blocks['ATSI_7X8'] gives 1 e.g.
        power_gas_block_idx = {pg_name: pg_idx for pg_name, pg_idx in
                               zip(power_gas_blocks, range(len(power_gas_blocks)))}

        power_gas_cash_vol_names = set([item
                                        for sublist in power_blocks_names
                                        for item in sublist])  # different blocks
        power_gas_cash_vol_names.add(fuel_idx_name)

        for days_bl, days_cv, fuel_cv in zip(power_blocks_names,
                                             cash_vols['power'],
                                             cash_vols['fuel']):
            for hour_bl, hour_cv, hour_fuel_cv in zip(days_bl, days_cv, fuel_cv):
                pg_idx = power_gas_block_idx[hour_bl]
                if not cash_vols_overwrite:
                    adj_fwd_tenors, adj_vol_tenors = mrds.find_adj_tenors(pg_idx, cash_fwd_tenors_days,
                                                                          cash_vol_tenors_days)
                    fwd_vol_tenors_numeric, fwd_vol_values_unexpired, fwd_vol_tenors_code, \
                    fwd_vol_tenors = ds.get_fwd_vol_curve_numeric_tenor(hour_cv, self.calcDate, self.calcDate,
                                                                        fwd_vol_ind='vol',
                                                                        adj_fwd_tenors_days=adj_fwd_tenors,
                                                                        adj_vol_tenors_days=adj_vol_tenors)
                    cash_vol_write = np.array(fwd_vol_values_unexpired[:(nb_fwds + 1)],
                                              dtype=np.double)
                    self.powerModels.set_cash_vols(pg_idx, cash_vol_write)
                else:
                    self.powerModels.set_cash_vols(pg_idx, cash_vols)

        fuel_com_nb = self.powerModels.nb_assets - 1
        adj_fwd_tenors, adj_vol_tenors = mrds.find_adj_tenors(fuel_com_nb, cash_fwd_tenors_days,
                                                              cash_vol_tenors_days)
        fwd_vol_tenors_numeric, fwd_vol_values_unexpired, fwd_vol_tenors_code, \
        fwd_vol_tenors = ds.get_fwd_vol_curve_numeric_tenor(cash_vols['fuel'][0][0], self.calcDate, self.calcDate,
                                                            fwd_vol_ind='vol',
                                                            adj_fwd_tenors_days=adj_fwd_tenors,
                                                            adj_vol_tenors_days=adj_vol_tenors)
        cash_vols_fuel = np.array(fwd_vol_values_unexpired[:(nb_fwds + 1)], dtype=np.double)
        power_models.set_cash_vols(power_gas_block_idx[fuel_idx_name], cash_vols_fuel)
        if manual_adj is not None:
            exec(manual_adj)
        return self.tolling_power_fuel_process_reduced( nb_sim,
                                                  days_partition,
                                                  hours_partition,
                                                  fuel_idx_name,
                                                  power_blocks_names,
                                                  power_models,
                                                  power_gas_block_idx,
                                                  nb_days,
                                                  fixed_monthly=fixed_monthly,
                                                  parallel=parallel,
                                                  cuda_ind=cuda_ind), power_gas_block_idx

    def tolling_power_fuel_process_reduced( self
                                          , nb_sim,
                                           fuel_idx_name,
                                           power_bl_names, power_models,
                                           power_gas_block_idx,
                                           cuda_ind=False):

        fwd_tenors_dt = power_models.forward_tenors_dt_list[0]
        tenors_chosen    = range(max(np.sum([ft < self.toll_start for ft in fwd_tenors_dt]) - 1, 0)
                                , max(np.sum([ft < self.toll_end   for ft in fwd_tenors_dt]) - 1, 0) + 1)

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

    @property
    def powerModels(self):
        """
        Constructs the ComSkew model.

        """

        if self.__powerModels:
            return self.__powerModels

        power_gas_blocks = self.power_gas_blocks

        self.__powerModels = mrds_utils.mrds_calib_multiple(list(power_gas_blocks),
                                             self.calcDate,
                                             [nb_fwds + 1] * len(power_gas_blocks),
                                             model_ind=model_ind,
                                             adj_fwd_tenors_days=adj_fwd_tenors_days,
                                             adj_vol_tenors_days=adj_vol_tenors_days)

    @powerModels.setter
    def powerModels(self, newPowerModels):
        self.__powerModels = newPowerModels

    @property
    def corrMatrix(self):
        """
        Correlation matrix of the power model.

        """

        return self.powerModels.complete_corr_mat

    @corrMatrix.setter
    def corrMatrix(self, new_corr_mtx):
        """
        Setting the power correlation matrix.

        """

        self.powerModels.complete_corr_mat = new_corr_mtx


    def generate_spots(self, m):
        """
        generates spots from month m

        :param m: month in the tolling process.
        :type m: int
        """

        days_tuple = (self.days_toll, self.days_d_toll, self.days_diff_toll, self.days_diff_l_toll)
        spot_blocks_m = [self.powerModels.simulate_spot_blocks_from_fom(self.fom_sims_all,
                                                                         asset_nb,
                                                                         m, self.nb_sim,
                                                                         self.days_partition,
                                                                         self.hours_partition,
                                                                         days_tuple,
                                                                         tenors_chosen=self.tenors_chosen,
                                                                         cuda_ind=self.cuda_ind)
                         for asset_nb in range(self.powerModels.nb_assets)]

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

    def cmgStartup(self
                  , cs
                  , ci = False ):
        """
        Startup decision of the cmg mode.

        """

        if not ci:
            cs['can_start'] = (cs.total_starts < self.powerPlantParams['maxMonthlyStarts']) & \
                              (cs.hours_shut >= self.powerPlantParams['minDownTime'])
        else:  # this kernel implements exactly what is above 3 lines
            cs['can_start'] = cuda_ops.comp_two_arrays_and(cs.total_starts
                                                           , cs.hours_shut
                                                           , self.powerPlantParams['maxMonthlyStarts']
                                                           , self.powerPlantParams['minDownTime'])

    def peakOnlyStartup(self
                       , cs
                       , ci = False ):
        """
        Startup only at peak times.

        """

        cnd_1 = cs.total_starts < self.powerPlantParams['maxMonthlyStarts']
        cnd_2 = cs.block_name == 'peak'
        cnd_3 = cs.hours_shut >= self.powerPlantParams['minDownTime']

        cs['can_start'] = cnd_1 & cnd_2 & cnd_3 if not ci else cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def offpeakOnlyStartup(self
                          , cs
                          , ci = False):
        """
        Startup only at peak times.

        """

        cnd_1 = cs.total_starts < self.powerPlantParams['maxMonthlyStarts']
        cnd_2 = cs.block_name != 'peak'
        cnd_3 = cs.hours_shut >= self.powerPlantParams['minDownTime']

        cs['can_start'] = cnd_1 & cnd_2 & cnd_3 if not ci else cuda_ops.min_int_three_cons(cnd_1, cnd_2, cnd_3)

    def startup_decision( self
                        , cs
                        , dispatch_mode = 'cmg'
                        , ci            = False):
        """
        Decision whether to start up.

        :param cs: current state
        :type cs: TODO: INSERT HERE.
        :param params: additional parameters to make the decision
        :param dispatch_mode: which type of dispatch would one want.
        :type dispatch_mode: str
        :param ci: indicator of whether cuda is used
        :type ci: bool
        """

        startup = { 'cmg'         : self.cmgStartup
                  , 'peak_only'   : self.peakOnlyStartup
                  , 'offpeak_only': self.offpeakOnlyStartup }

        startup[dispatch_mode](cs, self.powerPlantParams, ci)

    def forced_startup(self
                       , cs
                       , powerPrices
                       , fuelPrices
                       , dv=None
                       , dispatch_mode='cmg'
                       , ci=False):
        """
        Try to force power plant to start

        :param cs: current state object
        :param powerPrices: power prices
        :param fuelPrices: fuel prices TODO: HERE THE TYPE
        :param dispatch_mode: which dispatch to follow
        :param ci: cuda indicator (True or False)
        """

        if dispatch_mode == 'mrg':
            decision_1 = powerPrices - self.powerPlantParams['hrAtMax'] * fuelPrices
            cs.force_start = 2 * (decision_1 > dv[1]) + (decision_1 > dv[0]) & (decision_1 < dv[1])
        elif dispatch_mode == 'peak_only':
            cs.force_start.fill(2 * (cs.block_name == 'peak'))
        elif dispatch_mode == 'offpeak_only':
            cs.force_start.fill(2 * (cs.block_name != 'peak'))

    def cmgShutdown(self, cs, ci=False):
        """

        """

        if not ci:
            cs['can_shut'] = cs['hours_run'] >= self.powerPlantParams['minRunTime']
        else:
            cs['can_shut'] = cuda_ops.comp_array_number(cs['hours_run'], self.powerPlantParams['minRunTime'], op='larger', dtype='int32')

    def peakOnlyShutdown(self, cs, ci=False):
        cnd_2 = cs['block_name'] != 'peak'
        cnd_3 = cs['hours_run'] >= self.powerPlantParams['minRunTime']
        if not ci:
            cs.can_shut = cnd_2 & cnd_3
        else:
            cnd_3_int = cnd_3.astype(np.int32)  # TODO: THIS IS BAD Here!!!!
            cuda_ops.min_int_two(cnd_2, cnd_3_int, cs.can_shut)

    def offpeakOnlyShutdown(self, cs, ci=False):

        cnd_2 = cs['block_name'] == 'peak'
        cnd_3 = cs['hours_run']  >= self.powerPlantParams['minRunTime']

        if not ci:
            cs['can_shut'] = cnd_2 & cnd_3
        else:
            cuda_ops.min_int_two(cnd_2, cnd_3, cs['can_shut'])

    def shutdown_decision(self, cs, dispatch_mode='cmg', ci=False):
        """
        Decision whether it is sensible to shut down

        :param cs: current state, cs.can_shut is filled by this routine
           cs.can_shut is array of bools

        """

        { 'cmg'         : self.cmgShutdown
        , 'peak_only'   : self.peakOnlyShutdown
        , 'offpeak_only': self.offpeakOnlyShutdown }[dispatch_mode](cs, self.powerPlantParams, ci=ci)

    def forced_shutdown(self, cs, pp, fp, dv=None, dispatch_mode='cmg', ci=False):
        """
        decision to forcefully shut down, can take 3 outcomes: 2, 1, 0
        """
        if dispatch_mode == 'mrg':
            decision_1 = pp - self.powerPlantParams['hrAtMax'] * fp
            cs.force_shut = 2 * (decision_1 < dv[3]) + (decision_1 > dv[2]) & (decision_1 < dv[3])
        elif dispatch_mode == 'peak_only':
            cs.force_shut.fill(2 * (cs.block_name != 'peak'))
        elif dispatch_mode == 'offpeak_only':
            cs.force_shut.fill(2 * (cs.block_name != 'offpeak'))

    def set_other_params( self
                        , cs
                        , pl
                        , dispatch_mode='cmg'
                        , ci=False):
        """
        set up the cs.force_ and cs.can_ parameters

        :param pl: number of simulations for the month

        """

        np_gpa = gpa if self.cuda_ind else np

        if dispatch_mode == 'cmg' or dispatch_mode == 'mrg':

            if not ci:
                cs.force_start = np.ones(pl, dtype=np.short)
                cs.force_shut  = np.ones(pl, dtype=np.short)
                cs.can_start   = np.empty(pl, dtype=np.short)  # CHECK THIS ONE
                cs.can_shut    = np.empty(pl, dtype=np.short)
            else:  # cuda
                cs.force_start = gpa.empty(pl, dtype=np.int32).fill(1)
                cs.force_shut  = gpa.empty(pl, dtype=np.int32).fill(1)
                cs.can_start   = gpa.empty(pl, dtype=bool)
                cs.can_shut    = gpa.empty(pl, dtype=bool)
        elif dispatch_mode == 'always_run':
            if not ci:  # cpu
                cs.force_start = np.empty(pl, dtype=np.short).fill(2)
                cs.force_shut  = np.zeros(pl, dtype=np.short)
                cs.can_start   = np.empty(pl, dtype=np.short).fill(1)
                cs.can_shut    = np.empty(pl, dtype=np.short).fill(0)
            else:  # cuda
                cs.force_shut  = gpa.empty(dtype=np.int32).fill(0)  # force shut done once
                cs.force_start = gpa.empty(dtype=np.int32).fill(2)  # force start set here
                cs.can_shut    = gpa.zeros(pl, dtype=bool)
                cs.can_start   = gpa.zeros(pl, dtype=bool) + 1
        else:  # peak & offpeak only
            if not ci:  # cpu
                cs.force_start = np.empty(pl, dtype=np.short)
                cs.force_shut  = np.empty(pl, dtype=np.short)
                cs.can_start   = np.empty(pl, dtype=np.short)
                cs.can_shut    = np.empty(pl, dtype=np.short)
            else:  # cuda
                cs.force_start = 1  # TODO: FIX FIX FIX FIX
                cs.force_shut  = 1
                cs.can_start   = 1
                cs.can_shut    = 1

    def __blockDispatch( self
                       , spot_idx
                       , powerPrices
                       , fuelPrices
                       , startupSPin
                       , cs     : dict
                       , nbSims : int
                       , opdDispatchFct ):
        """
        Dispatch a single block, changes the current state as appropriate.

        :param cs: current state.
        :param nbSims: number of simulations
        :param opdDispatchFct: function for one-period dispatch algorithm

        """

        opdDispatchFct( spot_idx
             , powerPrices
             , fuelPrices
             , self.powerPlantParams
             , startupSPin
             , cs['state'],
              cs['hours_in_state'],
              cs['generation'],
              cs['total_starts'],
              cs['hours_shut'],
              cs['hours_run'],
              cs['global_starts'],
              cs['can_start'],
              cs['can_shut'],
              cs['force_start'],
              cs['force_shut'],
              cs['hours_block'],
              cs['df'],
              nbSims,
              cf_per_path_tmp,
              cs )

    def dispatch_month( self
                      , m : int
                      , conseq_hours
                      , conseq_block_names
                      , pl
                      , power_spots
                      , fuel_spots
                      , dv            = None
                      , dispatch_mode = 'cmg'
                      , floatType     = np.double ):
        """
        Calculate the dispatch for month m

        :param m: month number
        :param dv: decision variable, for optimization

        :param dispatch_mode: dispatch algorithm - can be 'cmg', 'mrg', or 'always run'
        """

        np_gpa = gpa if self.cuda_ind else np

        cf_per_path_tmp    = np_gpa.empty(pl, dtype=floatType)  # cash flow per path
        cf_per_path        = np_gpa.zeros(pl, dtype=floatType)  # cumulative cash flows

        cs = {'dispatch_mode':  dispatch_mode}
        self.set_other_params(cs, pl, dispatch_mode=dispatch_mode, ci=self.cuda_ind)
        cs['df'] = 1.

        # one period dispatch function
        opd_f = opd_1fuel.opd_1fuel if not self.cuda_ind else opd_1fuel_cu.opd_kernel

        for spot_idx, (block_hours, block_name) in enumerate(zip(conseq_hours, conseq_block_names)):
            power_prices, fuel_prices = power_spots[spot_idx, :], fuel_spots[spot_idx, :]

            # new state update
            cs.update( { 'hours_block'   : block_hours
                       , 'block_name'    : block_name } )

            if dispatch_mode == 'cmg':
                start_cost_init = self.powerPlantParams['fixedStartupCostCold'] + \
                                  self.powerPlantParams['SF'] * (fuel_prices + self.powerPlantParams['addFuelCost']) - \
                                  self.powerPlantParams['E_S'] * power_prices
                # startup shadow prices
                startup_sp_in = start_cost_init / (self.powerPlantParams['maxCap'] * self.powerPlantParams['startupHorizon'])
            else:
                startup_sp_in = 0.

            self.powerPlantParams['startupSPin'] = startup_sp_in

            # the reason why first are overwritten is that they change very often
            self.startup_decision (cs, self.powerPlantParams, dispatch_mode=dispatch_mode, ci=self.cuda_ind)
            self.shutdown_decision(cs, self.powerPlantParams, dispatch_mode=dispatch_mode, ci=self.cuda_ind)

            # the following updates cs.force_shut, cs.force_start
            self.forced_startup (cs, power_prices, fuel_prices, dv, dispatch_mode=dispatch_mode, ci=self.cuda_ind)
            self.forced_shutdown(cs, power_prices, fuel_prices, dv, dispatch_mode=dispatch_mode, ci=self.cuda_ind)

            # dispatch for a block.
            self.__blockDispatch( spot_idx
                                , power_prices
                                , fuel_prices
                                , startup_sp_in
                                , cs
                                , len(power_prices)  # number of simulations
                                , opd_f)

            cf_per_path += cf_per_path_tmp

        return self.powerModels._discount_discount[m] * cf_per_path

    def dispatchAll(self, dispatch_mode='cmg'):
        """
        Dispatches the algorithm for all months

        """

        months_to_compute = self.tenors_chosen
        conseq_hours, conseq_block_names = TollingModel.construct_consequitive_hours(self.days_partition,
                                                                                          self.hours_partition,
                                                                                          self.nb_days,
                                                                                          self.hours_partition_names)

        dispatchResult = {}
        for m in months_to_compute:
            ps, fs = self.generate_spots(m)
            dispatchResult[m] = self.dispatch_month( m
                                                   , conseq_hours
                                                   , conseq_block_names
                                                   , self.nb_sim
                                                   , ps
                                                   , fs
                                                   , dispatch_mode=dispatch_mode)

        if not self.cuda_ind:
            dispatch_res_months = {m: np.mean(dispatchResult[m]) for m in months_to_compute}
        else:
            dispatch_res_months = {m: gpa.sum(dispatchResult[m])/dispatchResult[m].size for m in months_to_compute}

        return { 'cashflow_by_month': dispatch_res_months
               , 'cashflow_total'   : sum(dispatch_res_months.values()) }

    def resimulate_prices(self):
        resimulating = self.tolling_power_fuel_process_reduced(self.toll_start, self.toll_end,
                                                          self.nb_sim,
                                                          self.days_partition,
                                                          self.hours_partition,
                                                          self.fuel_idx_name,
                                                          self.power_blocks_names,
                                                          self.power_models,
                                                          self.power_gas_block_idx,
                                                          self.nb_days)
        self.power_spots = resimulating['power_spot']
        self.fuel_spots  = resimulating['fuel_spot']
