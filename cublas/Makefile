# swig the cublas.h file

INCLUDE_DIR=-I/usr/include/python3.5/
LIBS_CUBLAS=-lcublas
LIBS_CUDA=-lcudart -lcuda
LIBS_CURAND=-lcurand
LIB_PYTHON=-lpython3.5m

# cuda specific 
PTXAS_OPTIONS=--ptxas-options=-v
CUDA_COMPILER_OPTS=--compiler-options '-fPIC'
CUDA_LIBS=-L/usr/lib/gcc/x86_64-linux-gnu/5/

# cublas section 
# performs all three steps in one 
cublas: cublas.i cublas_wrap.c 
	swig -python cublas.i
	gcc -g -fPIC -c $(INCLUDE_DIR) cublas_wrap.c -o cxx/cublas_i.o
	ld $(LIBS_DIR) -shared cxx/cublas_i.o $(LIBS_CUBLAS) $(LIBS_CUDA) $(LIB_PYTHON) -o _cublas_i.so

# curand section 
curand: curand.i curand_wrap.c
	swig -python curand.i
	gcc -fPIC -c $(INCLUDE_DIR) curand_wrap.c -o cxx/curand.o
	ld $(LIBS_DIR) -shared cxx/curand.o $(LIBS_CURAND) $(LIBS_CUDA) $(LIB_PYTHON) -o _curand.so


# kronecker lib
kronecker: kronecker_gpu.i kronecker_gpu_wrap.c 
	nvcc -c $(PTXAS_OPTIONS) $(CUDA_COMPILER_OPTS)  $(INCLUDE_DIR) $(LIBS_DIR) kronecker_gpu.cu -o cxx/kronecker_gpu.o
	swig -python kronecker_gpu.i
	gcc -fPIC -c $(INCLUDE_DIR) kronecker_gpu_wrap.c -o cxx/kronecker_gpu_wrap.o
	ld -shared $(LIBS_DIR) $(CUDA_LIBS) $(LIBS_CUDA) $(LIB_PYTHON) -lstdc++ cxx/kronecker_gpu.o cxx/kronecker_gpu_wrap.o -o _kronecker_gpu.so
