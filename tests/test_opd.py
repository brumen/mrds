import numpy as np
import pycuda.gpuarray as gpa

from unittest import TestCase

from opd import opd_avx
from opd.opd_1fuel_cu import getOpdHeader


class TestOpd(TestCase):

    N = 128000
    r = np.random.rand(N)
    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)
    d = np.random.rand(N)
    y = np.empty(N)

    def sampleInputs(self):
        """
        Sample cuda params.

        """

        sampleParams = ( 6.     # hrAtMax
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

        return sampleParams

    def test_just_run(self):
        """
        Tests whether add4 function works correctly.

        """

        y1 = np.empty(TestOpd.N)
        opd_avx.add4(TestOpd.r, TestOpd.a, TestOpd.b, TestOpd.c, TestOpd.d, y1, TestOpd.N)

        self.assertEqual( y1
                        , TestOpd.r - (TestOpd.a + TestOpd.b + TestOpd.c + TestOpd.d))

    def test_opd_cuda_1(self):
        """
        Tests whether the opd cuda even runs.
        """

        N = 100024

        sampleParams = (6.  # hrAtMax
                        , 7.  # hrAtMin
                        , 1000.  # maxCap - maximum capacity
                        , 100.  # minDisp - minimum dispatch
                        , 10.  # startFuel - startup fuel
                        , 15.  # startFuelCold - startup fuel from cold.
                        , 5.  # addFuelCost - added fuel costs
                        , 10.  # VC - variable costs
                        , 3.  # rampRate - ramp rate
                        , 0.1  # shutdownSPin - shutdown shadow price in
                        , 8.  # minDownTime - minimum downtime
                        , 16.  # minRunTime - minimum run time.
                        , 10.  # fixedStartupCost
                        , 5.  # maxMonthlyStarts
                        , 10.  # coldStartup
                        , 16.  # startupHorizon
                        , 16.  # shutdownHorizon
                        , 10.  # rampUpSPin
                        , 10.  # rampDownSPin
                        , 10.  # rampUpCost
                        , 10.  # rampDownCost
                        , 15.  # rampUpHorizon
                        , 25.  # rampDownHorizon
                        )

        powerPrices = np.random.rand(N)
        fuelPrices = np.random.rand(N)
        startupSPin = np.random.rand(N)
        stateState = np.random.binomial(2, 0.5, N)
        hoursInState = np.random.binomial(10, 0.5, N)
        stateGeneration = np.random.rand(N) * 10
        stateTotalStarts = np.random.binomial(3, 0.5, N)
        stateHoursShut = np.random.binomial(3, 0.5, N)
        stateHoursRun = np.random.binomial(3, 0.5, N)
        stateGlobalStarts = np.random.binomial(3, 0.5, N)
        dcCanStart = np.random.binomial(2, 0.5, N)
        dcCanShut = np.random.binomial(2, 0.5, N)
        dcForceStart = np.random.binomial(2, 0.5, N)
        dcForceShut = np.random.binomial(2, 0.5, N)

        opd_kernel = getOpdHeader()
        float_type = np.double

        for i in range(10000):
            opd_kernel(np.intc(0)  # block number
                       , gpa.to_gpu(powerPrices).astype(float_type)  # FLOAT_TYPE *power prices
                       , gpa.to_gpu(fuelPrices).astype(float_type)  # FLOAT_TYPE *fuel prices
                       , gpa.to_gpu(np.array(sampleParams)).astype(float_type)  # FLOAT_TYPE * opdParams,
                       , gpa.to_gpu(startupSPin).astype(float_type)  # FLOAT_TYPE * startupSPin,
                       , gpa.to_gpu(stateState).astype(np.bool)  # bool * state_state,
                       , gpa.to_gpu(hoursInState).astype(np.intc)  # int * state_hoursInState,
                       , gpa.to_gpu(stateGeneration).astype(float_type)  # FLOAT_TYPE * state_Generation,
                       , gpa.to_gpu(stateTotalStarts).astype(np.intc)  # int * state_TotalStarts,
                       , gpa.to_gpu(stateHoursShut).astype(np.intc)  # int * state_hoursShut,
                       , gpa.to_gpu(stateHoursRun).astype(np.intc)  # int * state_hoursRun,
                       , gpa.to_gpu(stateGlobalStarts).astype(np.intc)  # int * state_globalStarts,
                       , gpa.to_gpu(dcCanStart).astype(np.bool)  # bool * dc_canStart,
                       , gpa.to_gpu(dcCanShut).astype(np.bool)  # bool * dc_canShut,
                       , gpa.to_gpu(dcForceStart).astype(np.intc)  # int * dc_forceStart,
                       , gpa.to_gpu(dcForceShut).astype(np.intc)  # int * dc_forceShut,
                       , np.intc(8)  # int hours_in_block,
                       , np.double(0.99)  # FLOAT_TYPE df
                       , np.ulonglong(N)  # int nb_paths,
                       , block=(100, 1, 1)
                       , grid=(65000, 1))  # TODO: FIX THESE HERE!!!

        self.assertTrue(True)
