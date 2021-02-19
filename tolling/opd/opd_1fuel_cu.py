# elementwise kernel for one_period dispatch

import os

from pycuda.compiler import SourceModule

from mrds.config import work_dir, opd_1_fuel_cuda_code, CUDA_PRESENT

if CUDA_PRESENT:
    import pycuda.autoinit  # DO NOT REMOVE: THIS NEEDS TO BE HERE


def one_period_dispatch(float_type ='double', int_type ='int'):
    """ Returns the one period dispatch (OPD) function called opd_kernel.
        This kernel is used in the following way:

    opd_kernel(np.intc(0)  # block number
           , gpa.to_gpu(power_prices)           # FLOAT_TYPE *power prices
           , gpa.to_gpu(fuel_prices)            # FLOAT_TYPE *fuel prices
           , gpa.to_gpu(np.array(sampleParams)) # FLOAT_TYPE
           , gpa.to_gpu(startup_shadow_price)   # FLOAT_TYPE
           , gpa.to_gpu(stateState)             # bool
           , gpa.to_gpu(hoursInState)           # int
           , gpa.to_gpu(stateGeneration)        # FLOAT_TYPE
           , gpa.to_gpu(stateTotalStarts)       # int
           , gpa.to_gpu(stateHoursShut)         # int
           , gpa.to_gpu(stateHoursRun)          # int
           , gpa.to_gpu(stateGlobalStarts)      # int
           , gpa.to_gpu(dcCanStart)             # bool
           , gpa.to_gpu(dcCanShut)              # bool
           , gpa.to_gpu(dcForceStart)           # int
           , gpa.to_gpu(dcForceShut)            # int
           , np.intc(8)                         # intc
           , np.double(0.99)                    # FLOAT_TYPE, discount factor
           , np.ulonglong(N)                    # int
           , block=(100, 1, 1)
           , grid=(100, 1))

    :param float_type: type of float used on the GPU (default double)
    :param int_type: type of int used on the GPU (default int)
    """

    with open(os.path.join(work_dir, 'tolling', 'opd', opd_1_fuel_cuda_code), 'r') as opd_complete:
        #return SourceModule(opd_complete.read().replace('FLOAT_TYPE', float_type)).get_function('opd_kernel')
        return SourceModule(opd_complete.read()).get_function('opd_kernel')
