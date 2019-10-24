# elementwise kernel for one_period dispatch

import os

from pycuda.compiler import SourceModule

from config import work_dir, opd_1_fuel_cuda_code


def get_opd_module(float_type ='double', int_type ='int'):
    """ Returns the one period dispatch (OPD) function called opd_kernel.
        This kernel is used in the following way:

    opd_kernel(np.intc(0)  # block number
           , gpa.to_gpu(power_prices).astype(float_type)  # FLOAT_TYPE *power prices
           , gpa.to_gpu(fuel_prices).astype(float_type)  # FLOAT_TYPE *fuel prices
           , gpa.to_gpu(np.array(sampleParams)).astype(float_type)  # FLOAT_TYPE * opdParams,
           , gpa.to_gpu(startup_shadow_price).astype(float_type)  # FLOAT_TYPE * startup_shadow_price,
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

    :param float_type: type of float used on the GPU (default double)
    :param int_type: type of int used on the GPU (default int)
    """

    with open(os.path.join(work_dir, 'opd', opd_1_fuel_cuda_code), 'r') as opd_complete:
        return SourceModule(opd_complete.read().replace('FLOAT_TYPE', float_type)).get_function('opd_kernel')
