# elementwise kernel for one_period dispatch

import os

import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpa

from config import work_dir, opd1FuelComplete


def getOpdHeader(floatType = 'double', intType = 'int'):
    """
    Returns the one period dispatch (OPD) function called opd_kernel.
    This kernel is used in the following way:

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
           , grid=(100, 1))

    :param floatType: type of float used on the GPU (default double)
    :param intType: type of int used on the GPU (default int)

    """

    with open(os.path.join(work_dir, 'opd', opd1FuelComplete), 'r') as opdComplete:
        return SourceModule( opdComplete.read().replace('FLOAT_TYPE', floatType))\
                           .get_function('opd_kernel')
