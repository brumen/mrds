# configuration file
CUDA_PRESENT = False

import sys

# adding various paths
prod_dir = '/home/brumen/prasic/work/mrds/'
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
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.curandom import rand as curand
    from pycuda.compiler import SourceModule
