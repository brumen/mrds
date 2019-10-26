import numpy as np
import pycuda.gpuarray as gpa

from unittest import TestCase

from tolling.opd              import opd_avx
from tolling.opd.opd_1fuel_cu import get_opd_module


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
        """
        Tests whether the opd cuda even runs.
        """

        N = 100024
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

        opd_module = get_opd_module()

        for i in range(10000):
            opd_module( np.intc(0)  # block number
                      , gpa.to_gpu(power_prices)  # FLOAT_TYPE
                       , gpa.to_gpu(fuel_prices)  # FLOAT_TYPE
                       , gpa.to_gpu(np.array(sample_params))  # FLOAT_TYPE
                       , gpa.to_gpu(startup_sp_in)  # FLOAT_TYPE
                       , gpa.to_gpu(state_state)  # bool
                       , gpa.to_gpu(hours_in_state)  # int
                       , gpa.to_gpu(state_generation)  # FLOAT_TYPE
                       , gpa.to_gpu(state_total_starts)  # np.intc
                       , gpa.to_gpu(state_hours_shut)  # np.intc
                       , gpa.to_gpu(state_hours_run)  # np.intc
                       , gpa.to_gpu(state_global_starts)  # int * state_globalStarts,
                       , gpa.to_gpu(dc_can_start)  # dtype = bool
                       , gpa.to_gpu(dc_can_shut)   # bool
                       , gpa.to_gpu(dc_force_start)  # np.intc
                       , gpa.to_gpu(dc_force_shut)  # np.intc
                       , np.intc(8)  # np.intc
                       , np.double(0.99)  # FLOAT_TYPE df
                       , np.ulonglong(N)  # int nb_paths,
                       , block=(100, 1, 1)
                       , grid=(65000, 1))  # TODO: FIX THESE HERE!!!

        self.assertTrue(True)
