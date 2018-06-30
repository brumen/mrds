# configuration file
import sys
from subprocess import Popen, PIPE

# cuda check
p = Popen(['prime-select', 'query'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()
# rc = p.returncode
if output == 'nvidia\n':
    CUDA_PRESENT = True  # True
else:
    CUDA_PRESENT = False

# adding various paths
prod_dir = '/home/brumen/work/mrds/'
work_dir = prod_dir
sys.path.append(work_dir)  # basic path
subdirs = ['cubl', 'cublas', 'tbbmc', 'tests', 'cva', 'cuda',
           'opd', 'tolling', 'quartic', 'tensor', 'vols', 'weather',
           'pricers', 'spikes', 'ao']
for sd in subdirs:
    sys.path.append(work_dir + sd)

# cython params
cython_include_dirs = []  # '/usr/local/lib/python2.7/dist-packages/numpy/core/include/']
cython_extra_link_args = []  # '-L/usr/local/lib/python2.7/dist-packages/numpy/core/lib']

if CUDA_PRESENT:  # cuda modules
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['--use_fast_math']
