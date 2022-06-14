# configuration file
from subprocess import Popen, PIPE

# cuda check
CUDA_PRESENT = False
# CUDA_PRESENT = True if output == b'nvidia\n' else False

if CUDA_PRESENT:
    output, _ = Popen( ['prime-select', 'query']
                     , stdin  = PIPE
                     , stdout = PIPE
                     , stderr = PIPE).communicate()
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['--use_fast_math']


# adding various paths
prod_dir = '/home/brumen/work/mrds/'
work_dir = prod_dir

# for One-period dispatch
opd_1_fuel_cuda_code = 'opd_1fuel_cu_by_block_new.c'

brumen_pass = 'c2D779Mu'
