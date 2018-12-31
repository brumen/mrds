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

    def test_opd_1fuel(self):
        """
        Tests the 1 fuel dispatch algorithm.

        """

        N = TestOpd.N

        powerPrices = np.random.rand(N)
        fuelPrices  = np.random.rand(N)
        startupSPin = np.random.rand(N)
        stateState  = np.random.binomial(2, 0.5, N).astype(np.bool)  # TODO: first argument is NOT RIGHT
        hoursInState= np.random.binomial(10, 0.5, N)
        stateGeneration = np.random.rand(N) * 10
        stateTotalStarts = np.random.binomial(3, 0.5, N)
        stateHoursShut   = np.random.binomial(3, 0.5, N)
        stateHoursRun    = np.random.binomial(3, 0.5, N)
        stateGlobalStarts= np.random.binomial(3, 0.5, N)
        dcCanStart = np.random.binomial(2, 0.5, N).astype(np.bool)
        dcCanShut = np.random.binomial(2, 0.5, N).astype(np.bool)
        dcForceStart = np.random.binomial(2, 0.5, N).astype(np.bool)
        dcForceShut = np.random.binomial(2, 0.5, N).astype(np.bool)
        hoursInBlock = np.random.binomial(5, 0.5, N)

        opd.opd_1fuel_cu.opd_kernel( 0  # block number
                  , powerPrices            # FLOAT_TYPE *power prices
                  , fuelPrices             # FLOAT_TYPE *fuel prices
                  , self.sampleInputs()    # FLOAT_TYPE * opdParams,
                  , startupSPin            # FLOAT_TYPE * startupSPin,
                  , stateState             # bool * state_state,
                  , hoursInState           # int * state_hoursInState,
                  , stateGeneration        # FLOAT_TYPE * state_Generation,
                  , stateTotalStarts       # int * state_TotalStarts,
                  , stateHoursShut         # int * state_hoursShut,
                  , stateHoursRun          # int * state_hoursRun,
                  , stateGlobalStarts      # int * state_globalStarts,
                  , dcCanStart             # bool * dc_canStart,
                  , dcCanShut              # bool * dc_canShut,
                  , dcForceStart           # int * dc_forceStart,
                  , dcForceShut            # int * dc_forceShut,
                  , hoursInBlock           # int hours_in_block,
                  , 0.99                   # FLOAT_TYPE df
                  , N             )        # int nb_paths,

        self.assertTrue(True)

    def test_2_opd(self):
        N = 10

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
        stateState = np.random.binomial(2, 0.5, N).astype(np.bool)  # TODO: first argument is NOT RIGHT
        hoursInState = np.random.binomial(10, 0.5, N)
        stateGeneration = np.random.rand(N) * 10
        stateTotalStarts = np.random.binomial(3, 0.5, N)
        stateHoursShut = np.random.binomial(3, 0.5, N)
        stateHoursRun = np.random.binomial(3, 0.5, N)
        stateGlobalStarts = np.random.binomial(3, 0.5, N)
        dcCanStart = np.random.binomial(2, 0.5, N).astype(np.bool)
        dcCanShut = np.random.binomial(2, 0.5, N).astype(np.bool)
        dcForceStart = np.random.binomial(2, 0.5, N).astype(np.bool)
        dcForceShut = np.random.binomial(2, 0.5, N).astype(np.bool)

        opd_kernel = getOpdHeader()
        float_type = np.double

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
                   , grid=(100, 1))  # TODO: FIX THESE HERE!!!

        self.assertTrue(True)
