# to compile xmm files 
SWIG = swig
PROC_DIR = -funroll-all-loops -falign-functions=16 -O3
PROC_DIR_AVX = -funroll-all-loops -falign-functions=32 -O3
XMM = $(PROC_DIR) -msse4a
AVX2 = $(PROC_DIR_AVX) -mavx2
PYTHON_INCLUDE = -I/usr/include/python3.5/
GCC = g++
OMP = -fopenmp 

# vector times matrix cpu avx part
vtpm_cpu_avx: vtpm_cpu.c vtpm_cpu.i
	$(SWIG) -python -o cxx/vtpm_cpu_wrap.c -I. -Wall vtpm_cpu.i 
	$(GCC) $(OMP) $(AVX2) $(PYTHON_INCLUDE) -fPIC -c vtpm_cpu.c -o cxx/vtpm_cpu.o
	$(GCC) $(OMP) $(AVX2) $(PYTHON_INCLUDE) -fPIC -S vtpm_cpu.c -o cxx/vtpm_cpu.s
	$(GCC) $(OMP) $(AVX2) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/vtpm_cpu_wrap.c -o cxx/vtpm_cpu_wrap.o
	$(GCC) -shared cxx/vtpm_cpu.o cxx/vtpm_cpu_wrap.o -lgomp -o _cxx/vtpm_cpu.so
