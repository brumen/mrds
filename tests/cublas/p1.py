# Pycuda simple test

import pycuda.autoinit  # this initializes the cuda, if it wasnt already initialized
from numpy import double
import pycuda.gpuarray as gpa
import cublas_i
import cublas 

A1 = gpa.zeros((10, 10), dtype=double) + 1.
d1 = gpa.zeros(10,       dtype=double) + 1.
y1 = gpa.empty(10,       dtype=double)

print("D1", d1, d1.dtype)
print("D", cublas_i.cublasIsamax2_d(d1.ptr, 10))
print("H", cublas_i.cublasSgemv_d(A1.ptr, d1.ptr, y1.ptr, 10, 10, 10))
print("H", y1)
print("II", cublas.cublasSgemv(A1, d1))
