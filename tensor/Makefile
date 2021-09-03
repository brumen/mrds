CONFIG_PROGRAM=python-config

ALL_LIBS=$(shell $(CONFIG_PROGRAM) --libs)
INCLUDES=$(shell $(CONFIG_PROGRAM) --includes) -I/usr/include/pthread.h
CFLAGS=$(shell $(CONFIG_PROGRAM) --cflags)
LDFLAGS=$(shell $(CONFIG_PROGRAM) --ldflags)
GCC=gcc

LIB_FILE=libtp.so

tensor_prod: tensor_prod.c
	$(GCC) -c -fPIC tensor_prod.c $(CFLAGS) -o objs/tensor_prod.o
	$(GCC) -shared objs/tensor_prod.o $(LDFLAGS) -Wl,-soname -o $(LIB_FILE)

tensor_prod_omp: tensor_prod_omp.c
	$(GCC) $(LDFLAGS) -fopenmp -shared -Wl,-soname,tp -fPIC -o tp.so tensor_prod_omp.c
