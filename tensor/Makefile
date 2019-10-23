CONFIG_PROGRAM=python3.5-config

ALL_LIBS=$(shell $(CONFIG_PROGRAM) --libs)
INCLUDES=$(shell $(CONFIG_PROGRAM) --includes) -I/usr/include/pthread.h
CFLAGS=$(shell $(CONFIG_PROGRAM) --cflags)
LDFLAGS=$(shell $(CONFIG_PROGRAM) --ldflags)

LIB_FILE=libtp.so

tensor_prod: tensor_prod.c
	gcc -c -fPIC tensor_prod.c $(CFLAGS) -o objs/tensor_prod.o
	gcc -shared objs/tensor_prod.o $(LDFLAGS) -Wl,-soname -o $(LIB_FILE)
