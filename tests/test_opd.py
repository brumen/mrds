import numpy as np
import pycuda.gpuarray as gpa
import mrds.config

if mrds.config.CUDA_PRESENT:
    import pycuda.autoinit  # IMPORTANT: This line HAS TO BE HERE

from unittest import TestCase

from mrds.tolling.opd              import opd_avx
from mrds.tolling.opd.opd_1fuel_cu import one_period_dispatch
from mrds.tolling.opd.opd_1fuel    import opd_1fuel


class TestOpd(TestCase):

    N = 128000
    r = np.random.rand(N)
    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)
    d = np.random.rand(N)
    y = np.empty(N)

    def sample_inputs(self):
        """ Sample cuda params.
        """

        return ( 6.     # hrAtMax
               , 7.     # hrAtMin
               , 1000.  # maxCap - maximum capacity
               , 100.   # minDisp - minimum dispatch
               , 10.    # startFuel - startup fuel
               , 15.    # startFuelCold - startup fuel from cold.
               , 5.     # addFuelCost - added fuel costs
               , 10.    # VC - variable costs
               , 3.     # rampRate - ramp rate
               , 0.1    # shutdownSPin - shutdown shadow price in
               , 8.     # minDownTime - minimum downtime
               , 16.    # minRunTime - minimum run time.
               , 10.    # fixedStartupCost
               , 10.    # fixedStartupCostCold
               , 5.     # maxMonthlyStarts
               , 10.    # coldStartup
               , 16.    # startupHorizon
               , 16.    # shutdownHorizon
               , 10.    # rampUpSPin
               , 10.    # rampDownSPin
               , 10.    # rampUpCost
               , 10.    # rampDownCost
               , 15.    # rampUpHorizon
               , 25.    # rampDownHorizon
               )

    def test_add4_function(self):
        """ Tests whether add4 function works correctly.
        """

        y1 = np.empty(TestOpd.N)
        opd_avx.add4(TestOpd.r, TestOpd.a, TestOpd.b, TestOpd.c, TestOpd.d, y1, TestOpd.N)

        np.testing.assert_array_equal( y1
                                     , TestOpd.r - (TestOpd.a + TestOpd.b + TestOpd.c + TestOpd.d))

    def test_opd_cuda_1(self):
        """ Tests whether the opd cuda even runs.
        """

        N = 1024  # number of paths, simulations of power/fuel/decision/state prices.
        sample_params = self.sample_inputs()

        power_prices        = np.random.rand(N)
        fuel_prices         = np.random.rand(N)
        startup_sp_in       = np.random.rand(N)
        state_state         = np.random.binomial(2, 0.5, N).astype(bool)
        hours_in_state      = np.random.binomial(10, 0.5, N).astype(np.intc)
        state_generation    = np.random.rand(N) * 10
        state_total_starts  = np.random.binomial(3, 0.5, N).astype(np.intc)
        state_hours_shut    = np.random.binomial(3, 0.5, N).astype(np.intc)
        state_hours_run     = np.random.binomial(3, 0.5, N).astype(np.intc)
        state_global_starts = np.random.binomial(3, 0.5, N).astype(np.intc)
        dc_can_start        = np.random.binomial(2, 0.5, N).astype(bool)
        dc_can_shut         = np.random.binomial(2, 0.5, N).astype(bool)
        dc_force_start      = np.random.binomial(2, 0.5, N).astype(np.intc)
        dc_force_shut       = np.random.binomial(2, 0.5, N).astype(np.intc)

        power_prices_gpu = gpa.to_gpu(power_prices)
        cashflow = gpa.empty_like(power_prices_gpu)

        opd_module = one_period_dispatch()

        opd_module( np.intc(0)  # block number
                  , power_prices_gpu                      # FLOAT_TYPE
                   , gpa.to_gpu(fuel_prices)              # FLOAT_TYPE
                   , gpa.to_gpu(np.array(sample_params))  # FLOAT_TYPE
                   , gpa.to_gpu(startup_sp_in)            # FLOAT_TYPE
                   # state variables
                   , gpa.to_gpu(state_state)              # bool
                   , gpa.to_gpu(hours_in_state)           # int (perhaps intc)
                   , gpa.to_gpu(state_generation)         # FLOAT_TYPE
                   , gpa.to_gpu(state_total_starts)       # np.intc
                   , gpa.to_gpu(state_hours_shut)         # np.intc
                   , gpa.to_gpu(state_hours_run)          # np.intc
                   , gpa.to_gpu(state_global_starts)      # int
                   # decision variables
                   , gpa.to_gpu(dc_can_start)             # dtype = bool
                   , gpa.to_gpu(dc_can_shut)              # bool
                   , gpa.to_gpu(dc_force_start)           # np.intc
                   , gpa.to_gpu(dc_force_shut)            # np.intc
                   , np.intc(8)  # np.intc
                   , np.double(0.99)  # FLOAT_TYPE df
                   , np.intc(N)  # int nb_paths,
                   , cashflow
                   , block=(100, 1, 1)
                   , grid=(65000, 1))
        print(cashflow)

        self.assertTrue(True)

    def test_opd_regression(self):
        """ Tests whether the opd cuda and opd on cpu produce the same results.
        """

        nb_paths       = 1024  # number of paths, simulations of power/fuel/decision/state prices.
        hours_in_block = 8
        sample_params = self.sample_inputs()

        power_prices        = np.random.rand(nb_paths)
        fuel_prices         = np.random.rand(nb_paths)
        startup_sp_in       = np.random.rand(nb_paths)
        state_state         = np.random.binomial(2, 0.5, nb_paths).astype(bool)
        hours_in_state      = np.random.binomial(10, 0.5, nb_paths).astype(np.intc)
        state_generation    = np.random.rand(nb_paths) * 10
        state_total_starts  = np.random.binomial(3, 0.5, nb_paths).astype(np.intc)
        state_hours_shut    = np.random.binomial(3, 0.5, nb_paths).astype(np.intc)
        state_hours_run     = np.random.binomial(3, 0.5, nb_paths).astype(np.intc)
        state_global_starts = np.random.binomial(3, 0.5, nb_paths).astype(np.intc)
        dc_can_start        = np.random.binomial(2, 0.5, nb_paths).astype(bool)
        dc_can_shut         = np.random.binomial(2, 0.5, nb_paths).astype(bool)
        dc_force_start      = np.random.binomial(2, 0.5, nb_paths).astype(np.intc)
        dc_force_shut       = np.random.binomial(2, 0.5, nb_paths).astype(np.intc)

        power_prices_gpu = gpa.to_gpu(power_prices)
        cashflow = gpa.empty_like(power_prices_gpu)

        opd_module = one_period_dispatch()

        opd_module( np.intc(0)  # block number
                  , power_prices_gpu                      # FLOAT_TYPE
                   , gpa.to_gpu(fuel_prices)              # FLOAT_TYPE
                   , gpa.to_gpu(np.array(sample_params))  # FLOAT_TYPE
                   , gpa.to_gpu(startup_sp_in)            # FLOAT_TYPE
                   # state variables
                   , gpa.to_gpu(state_state)              # bool
                   , gpa.to_gpu(hours_in_state)           # int (perhaps intc)
                   , gpa.to_gpu(state_generation)         # FLOAT_TYPE
                   , gpa.to_gpu(state_total_starts)       # np.intc
                   , gpa.to_gpu(state_hours_shut)         # np.intc
                   , gpa.to_gpu(state_hours_run)          # np.intc
                   , gpa.to_gpu(state_global_starts)      # int
                   # decision variables
                   , gpa.to_gpu(dc_can_start)             # dtype = bool
                   , gpa.to_gpu(dc_can_shut)              # bool
                   , gpa.to_gpu(dc_force_start)           # np.intc
                   , gpa.to_gpu(dc_force_shut)            # np.intc
                   , np.intc(hours_in_block)              # np.intc
                   , np.double(0.99)                      # FLOAT_TYPE
                   , np.intc(nb_paths)                    # int
                   , cashflow
                   , block=(100, 1, 1)
                   , grid=(65000, 1))
        print(cashflow)

        cashflow_cpu = np.empty_like(power_prices)
        curr_state = { 'state'         : state_state
                     , 'generation'    : state_generation
                     , 'hours_shut'    : state_hours_shut
                     , 'hours_in_state': hours_in_state
                     , 'hours_run'     : state_hours_run
                     , 'total_starts'  : state_total_starts
                     , 'global_starts' : state_global_starts }
        curr_decision = { 'can_start'  : dc_can_start
                        , 'can_shut'   : dc_can_shut
                        , 'force_start': dc_force_start
                        , 'force_shut' : dc_force_shut}

        # opd computed on the cpu:
        opd_1fuel( power_prices
                 , fuel_prices
                 , sample_params
                 , startup_sp_in
                 , curr_state
                 , curr_decision
                 , hours_in_block
                 , nb_paths
                 , cashflow_cpu)
        print(cashflow_cpu)

        print(1)
        np.testing.assert_almost_equal(cashflow.get(), cashflow_cpu)
