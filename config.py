# configuration file
import sys
from subprocess import Popen, PIPE

# cuda check
output, _ = Popen( ['prime-select', 'query']
                 , stdin  = PIPE
                 , stdout = PIPE
                 , stderr = PIPE).communicate()
CUDA_PRESENT = True if output == b'nvidia\n' else False

# adding various paths
prod_dir = '/home/brumen/work/mrds/'
work_dir = prod_dir
sys.path.append(work_dir)  # basic path

# cython params
cython_include_dirs    = []
cython_extra_link_args = []

if CUDA_PRESENT:  # cuda modules
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['--use_fast_math']

# for One-period dispatch
opd1FuelComplete = 'opd_1fuel_cu_by_block_new.c'
