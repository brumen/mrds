# module for multivariate minimization 
import config
from numpy import *
from numpy.random import rand
import scipy
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats 
import scipy.optimize 

import time


lattice_size = 100000000
x_h = rand(lattice_size)

fct_string = open("ndim_fct.cu").read()
fct_module = config.SourceModule(fct_string%{"lattice_size": lattice_size})
fct_apply = fct_module.get_function("f1")  # extracting compute vol function

t1 = time.time()
res1 = sin(x_h - 0.5) * cos(x_h-0.5)
t1 = time.time() - t1


def exec_apply(fct, x_d, x_size):
    if x_size < 512:
        fct(x_d, block=(x_size, 1, 1), grid=(1, 1))
    else:
        fct(x_d, block=(512, 1, 1), grid=(min((x_size+511)/512, 65535), 1))

t2 = time.time()
x_d = config.gpuarray.to_gpu(x_h.astype(float32))
for ind in range(50000):
    exec_apply(fct_apply, x_d, lattice_size)
x_d = x_d.get()
t2 = time.time() - t2

print "Difference = ", scipy.linalg.norm(res1 - x_d)
print "Time host = ", t1
print "Time dev = ", t2
print "Dev speedup = ", t1/t2
