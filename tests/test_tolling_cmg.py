import time
import tolling_cmg
import test_params
import mrds
import opd_1fuel
import opd_1fuel_cu
import pycuda
import pycuda.gpuarray as gpa
import numpy as np
import curand

from unittest import TestCase


class TestTolling(TestCase):

    def test_curand(self):
        """ tests the effectiveness of the curand generator
        """

        g1 = curand.create_gen_simple()
        for i in range(1000000):
            simulated_rn_init = gpa.empty((2, 50000), dtype=np.float32)
            curand.gen_eff_dev_rns(simulated_rn_init.size, np.longlong(simulated_rn_init.ptr), g1)

    def test_dispatch_alg(self):
        """ testing the function
        """

        nb = 2 ** 5
        cuda_ind = True
        spot_idx = 0
        power_prices = np.random.random(nb) * 30
        power_prices_d = gpa.to_gpu(power_prices.astype(np.float32))
        fuel_prices = np.random.random(nb) * 5
        fuel_prices_d = gpa.to_gpu(fuel_prices.astype(np.float32))
        params = test_params.test_simplest_toll_params()
        params_used = [params.hrAtMax,
                       params.hrAtMin,
                       params.maxCap,
                       params.minDisp,
                       params.startFuel,
                       params.startFuelCold,
                       params.addFuelCost,
                       params.VC,
                       params.rampRate,
                       params.shutdownSPin,
                       params.minDownTime,
                       params.minRunTime,
                       params.fixedStartupCost,
                       params.fixedStartupCostCold,
                       params.maxMonthlyStarts,
                       params.coldStartup,
                       params.startupHorizon,
                       params.shutdownHorizon,
                       params.rampUpSPin,
                       params.rampDownSPin,
                       params.rampUpCost,
                       params.rampDownCost,
                       params.rampUpHorizon,
                       params.rampDownHorizon]
        params_used_cuda = gpa.to_gpu(np.array(params_used)).astype(np.float32)

        nus_hours_in_state = np.zeros(nb, dtype=np.int16)
        nus_generation = np.zeros(nb)
        nus_total_starts = np.zeros(nb, dtype=np.short)
        nus_hours_shut = np.zeros(nb, dtype=np.short)
        nus_hours_run = np.zeros(nb, dtype=np.short) + 10
        nus_global_starts = np.zeros(nb, dtype=np.short)
        nus_state = np.empty(nb, dtype=np.short)  # initial state, not running
        nus_state.fill(-1)
        results = np.empty(nb)  # results lumped together
        startupSPin = np.zeros(nb) + 0.2

        cs = tolling_cmg.tolling_params()
        cs.df = 1.
        cs.hours_in_state = nus_hours_in_state
        cs.state = nus_state
        cs.generation = nus_generation
        cs.total_starts = nus_total_starts
        cs.hours_shut = nus_hours_shut
        cs.hours_run = nus_hours_run
        cs.global_starts = nus_global_starts
        cs.can_start = np.ones(nb, dtype=np.short) * (-1)
        cs.can_shut = np.ones(nb, dtype=np.short) * (-1)
        cs.force_start = np.ones(nb, dtype=np.short)
        cs.force_shut = np.ones(nb, dtype=np.short)
        cs.hours_block = 8

        nus_hours_in_state_d = gpa.zeros(nb, dtype=np.int32)  # 0
        nus_generation_d = gpa.zeros(nb, dtype=np.float32)  # 0
        nus_total_starts_d = gpa.zeros(nb, dtype=np.int32)   # 0
        nus_hours_shut_d = gpa.zeros(nb, dtype=np.int32) + 10  # 10
        nus_hours_run_d = gpa.zeros(nb, dtype=np.int32) + 10  # 10
        nus_global_starts_d = gpa.zeros(nb, dtype=np.int32)  # 0
        nus_state_d = gpa.empty(nb, dtype=bool)  # initial state, not running
        nus_state_d.fill(True)  # True
        results_d = gpa.empty(nb * 9, dtype=np.float32)  # results lumped together
        startupSPin_d = gpa.zeros(nb, dtype=np.float32) + 0.2  # 0.2
        curr_state = gpa.empty(nb, dtype=bool)
        # isp = gpa.empty(nb_sims, dtype=bool)
        # dsh = gpa.empty(nb_sims, dtype=bool)

        cs_d = tolling_cmg.tolling_params()
        cs_d.df = 1.
        cs_d.hours_in_state = nus_hours_in_state_d
        cs_d.state = nus_state_d
        cs_d.generation = nus_generation_d
        cs_d.total_starts = nus_total_starts_d
        cs_d.hours_shut = nus_hours_shut_d
        cs_d.hours_run = nus_hours_run_d
        cs_d.global_starts = nus_global_starts_d
        cs_d.can_start = gpa.empty(nb, dtype=bool)
        cs_d.can_start.fill(True)
        cs_d.can_shut = gpa.empty(nb, dtype=bool)
        cs_d.can_shut.fill(True)
        cs_d.force_start = gpa.zeros(nb, dtype=np.int32) + 1
        cs_d.force_shut = gpa.zeros(nb, dtype=np.int32) + 1
        cs_d.hours_block = 8

        for idx in range(100000):
            # print "IDX", idx
            if cuda_ind:
                opd_1fuel_cu.opd_k(spot_idx,
                                   power_prices_d, fuel_prices_d,
                                   params_used_cuda,  # params_used_cuda,
                                   startupSPin_d,
                                   cs_d.state,
                                   cs_d.hours_in_state,
                                   cs_d.generation,
                                   cs_d.total_starts,
                                   cs_d.hours_shut,
                                   cs_d.hours_run,
                                   cs_d.global_starts,
                                   cs_d.can_start,
                                   cs_d.can_shut,
                                   cs_d.force_start,
                                   cs_d.force_shut,
                                   cs_d.hours_block,
                                   cs_d.df,
                                   nb,
                                   nus_hours_in_state_d,
                                   nus_generation_d,
                                   nus_total_starts_d,
                                   nus_hours_shut_d,
                                   nus_hours_run_d,
                                   nus_global_starts_d,
                                   nus_state_d,
                                   results_d,
                                   curr_state)
            else:
                opd_1fuel.opd_1fuel(spot_idx,
                                    power_prices, fuel_prices,
                                    params_used,
                                    startupSPin,
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
                                    nb,
                                    nus_hours_in_state,
                                    nus_generation,
                                    nus_total_starts,
                                    nus_hours_shut,
                                    nus_hours_run,
                                    nus_global_starts,
                                    nus_state,
                                    results)

    def test_simplest_toll(self):
        cuda_ind = False
        params = test_params.test_simplest_toll_params()
        market = test_params.test_simplest_toll_market()
        days_block, hours_block, hours_block_names, days_block_names, power_bl_names,\
            fuel_idx_name, cash_vols = tolling_cmg.tolling_market_tuple(market)

        market_date = '20150401'
        toll_start = '20150501'
        toll_end = '20161231'
        nb_sim = 50000
        model_ind = 'skew'

        tolling_model = tolling_cmg.tolling_model_CMG(market_date,
                                                      toll_start, toll_end,
                                                      nb_sim,
                                                      power_bl_names,
                                                      fuel_idx_name,
                                                      days_block, days_block_names,
                                                      hours_block, hours_block_names,
                                                      cash_vols,
                                                      params,
                                                      model_ind=model_ind,
                                                      cuda_ind=cuda_ind)

        res = tolling_model.dispatch_cmg()

    def test_simplest_toll_2(self):
        cuda_ind = False
        nb_sim = 500
        params = test_params.test_simplest_toll_params()
        market = test_params.test_simplest_toll_market()
        days_block, hours_block, hours_block_names, days_block_names, power_bl_names,\
            fuel_idx_name, cash_vols = tolling_cmg.tolling_market_tuple(market)

        market_date = '20150401'
        toll_start = '20150501'
        toll_end = '20261031'
        # toll_end = '20181031'
        model_ind = 'skew'

        mm_new = mrds.mrds_calib_multiple(['ATSI_2X16',
                                           'ATSI-PEAK',
                                           'ATSI_7X8',
                                           'NG_MICHCON_GD-PEAK'],
                                           '20150401', [139, 139, 139, 139])
        mm_corr = mrds.mrds_calib_multiple(['ATSI_2X16',
                                            'ATSI-PEAK',
                                            'ATSI_7X8',
                                            'NG_MICHCON_GD-PEAK'],
                                            '20150401', [9, 9, 9, 9])
        mm_new.complete_corr_mat = mm_corr.complete_corr_mat

        tolling_model = tolling_cmg.tolling_model_CMG(market_date,
                                                      toll_start, toll_end,
                                                      nb_sim,
                                                      power_bl_names,
                                                      fuel_idx_name,
                                                      days_block, days_block_names,
                                                      hours_block, hours_block_names,
                                                      cash_vols,
                                                      params,
                                                      model_ind=model_ind,
                                                      cuda_ind=cuda_ind,
                                                      mm_overwrite=mm_new)

        res = tolling_model.dispatch_cmg()


    def gen_toll_object(cuda_ind=False, nb_sim=500):
        params = test_params.test_simplest_toll_params()
        market = test_params.test_simplest_toll_market()
        days_block, hours_block, hours_block_names, days_block_names, power_bl_names,\
            fuel_idx_name, cash_vols = tolling_cmg.tolling_market_tuple(market)

        market_date = '20150401'
        toll_start = '20150501'
        toll_end = '20261031'
        model_ind = 'skew'

        mm_new = mrds.mrds_calib_multiple(['ATSI_2X16',
                                           'ATSI-PEAK',
                                           'ATSI_7X8',
                                           'NG_MICHCON_GD-PEAK'],
                                           '20150401', [139, 139, 139, 139])
        mm_corr = mrds.mrds_calib_multiple(['ATSI_2X16',
                                            'ATSI-PEAK',
                                            'ATSI_7X8',
                                            'NG_MICHCON_GD-PEAK'],
                                            '20150401', [9, 9, 9, 9])
        mm_new.complete_corr_mat = mm_corr.complete_corr_mat
        tolling_model = tolling_cmg.tolling_model_CMG(market_date,
                                                      toll_start, toll_end,
                                                      nb_sim,
                                                      power_bl_names,
                                                      fuel_idx_name,
                                                      days_block, days_block_names,
                                                      hours_block, hours_block_names,
                                                      cash_vols,
                                                      params,
                                                      model_ind=model_ind,
                                                      cuda_ind=cuda_ind,
                                                      mm_overwrite=mm_new)

        # res = tolling_model.dispatch_cmg()
        return tolling_model

    # try it with:
    # test_simplest_toll(cuda_ind=True)

    def run_tests_1():
        t1 = time.time()
        test_simplest_toll_2(cuda_ind=True, nb_sim=50000)
        t1 = time.time()
        test_simplest_toll_2(cuda_ind=False, nb_sim=50000)
