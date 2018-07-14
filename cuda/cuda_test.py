import config
import time
import numpy as np
import pycuda.curandom
import pycuda.gpuarray as gpa
import pycuda.reduction
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel


# vector times vector - constructing a matrix, first vec is column, second is row
t1_code = open(config.work_dir + 'cuda/change9.c', 'r').read()
t1_module = SourceModule(t1_code)
t1_f = t1_module.get_function("t1")

def t1(n):
    m1_d = gpa.zeros((9,n), dtype=np.float32) + 1
    res = gpa.empty(n, dtype=np.float32)
    block_dims = (1, 1, 1)  # THIS HAS TO BE m_cols, 1, 1 & m_cols < 64
    grid_dims = (n, 1)
    t1 = time.time()
    for i in range(10000):
        t1_f(m1_d, res, np.int32(n), block=block_dims, grid=grid_dims)
    print "T1:", time.time() - t1
    return res


t2 = ElementwiseKernel("float *a1, float *a2, float *a3, float *a4, float *a5, float *a6, float *a7, float *a8, float *a9, float *res", 
                       "res[i] = a1[i] + a2[i] + a3[i] + a4[i] + a5[i] + a6[i] + a7[i] + a8[i] + a9[i]",
                       name="t2")


def t2_use(n):
    """
    usage in tolling_cmg startup decision
    """
    m9 = gpa.zeros((9, n), dtype=np.float32) + 1
    res = gpa.empty(n, dtype=np.float32)
    t1 = time.time()
    for i in range(10000):
        t2(m9[0,:], m9[1,:], m9[2,:], m9[3,:], m9[4,:], m9[5,:], m9[6,:], m9[7,:], m9[8,:], res)
    print "T2", time.time() - t1
    return res
