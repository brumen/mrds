# to compile xmm files
SWIG = swig
CPP  = g++
PROC_DIR     = -funroll-all-loops -falign-functions=16 -O3
PROC_DIR_AVX = -funroll-all-loops -falign-functions=32 -O3
XMM  = $(PROC_DIR) -msse4a
AVX2 = $(PROC_DIR_AVX) -mavx2
PYTHON_INCLUDE = $(shell python-config --includes)

# compilation abbrevation
CPP_XMM_COMPILE = $(CPP) $(XMM) $(PYTHON_INCLUDE)
CPP_AVX_COMPILE = $(CPP) $(AVX2) $(PYTHON_INCLUDE)


opd_xmm: opd_xmm.i opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i
	$(CPP_XMM_COMPILE) -fPIC -c opd_xmm.c -o cxx/opd_xmm.o
	$(CPP_XMM_COMPILE) -fPIC -S opd_xmm.c -o cxx/opd_xmm.s
	$(CPP_XMM_COMPILE) -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
	$(CPP) -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so


# debugging version of the above
opd_xmm_debug: opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i
	$(CPP) $(XMM) -g -fPIC -c opd_xmm.c -o cxx/opd_xmm.o
	$(CPP) $(XMM) -g -fPIC -S opd_xmm.c -o cxx/opd_xmm.s
	$(CPP) $(XMM) $(PYTHON_INCLUDE) -g -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
	$(CPP) -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so


opd_start_fom_S: cxx/opd_xmm_ch.s
	$(CPP) -fPIC $(XMM) -c cxx/opd_xmm_ch.s -o cxx/opd_xmm.o
	$(CPP) -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so


opd_1: opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i
	$(CPP) $(XMM) -fPIC -c cxx/opd_xmm_ch.s -o cxx/opd_xmm_ch.o
	$(CPP) $(XMM) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
	$(CPP) -shared cxx/opd_xmm_ch.o cxx/opd_xmm_wrap.o -o _opd_xmm.so


# avx part
opd_avx: opd_avx.i opd_avx.c
	$(SWIG) -python -o cxx/opd_avx_wrap.c -I. -Wall opd_avx.i
	$(CPP_AVX_COMPILE) -fPIC -c opd_avx.c -o cxx/opd_avx.o
	$(CPP_AVX_COMPILE) -fPIC -S opd_avx.c -o cxx/opd_avx.s
	$(CPP_AVX_COMPILE) -fPIC -E opd_avx.c -o cxx/opd_avx.pre
	$(CPP_AVX_COMPILE) -I. -fPIC -c cxx/opd_avx_wrap.c -o cxx/opd_avx_wrap.o
	$(CPP) -shared  cxx/opd_avx.o cxx/opd_avx_wrap.o -o _opd_avx.so

opd_avx_test:
	$(SWIG) -python -o cxx/opd_avx_wrap.c -I. -Wall opd_avx.i
	$(CPP_AVX_COMPILE) -fPIC -c opd_avx_test.c -o cxx/opd_avx_test.o
	$(CPP_AVX_COMPILE) -fPIC -S opd_avx.c -o cxx/opd_avx.s
	$(CPP_AVX_COMPILE) -fPIC -E opd_avx.c -o cxx/opd_avx.pre
	$(CPP_AVX_COMPILE) -I. -fPIC -c cxx/opd_avx_wrap.c -o cxx/opd_avx_wrap.o
	$(CPP) -shared  cxx/opd_avx.o cxx/opd_avx_wrap.o -o _opd_avx.so


# debugging version of the above
opd_avx_debug: opd_avx.i opd_avx.c
	$(SWIG) -python -o cxx/opd_avx_wrap.c -I. -Wall opd_avx.i
	$(CPP_AVX_COMPILE) -g -fPIC -c opd_avx.c -o cxx/opd_avx_g.o
	$(CPP_AVX_COMPILE) -g -fPIC -S opd_avx.c -o cxx/opd_avx_g.s
	$(CPP_AVX_COMPILE) -g -I. -fPIC -c cxx/opd_avx_wrap.c -o cxx/opd_avx_wrap_g.o
	$(CPP) -shared  cxx/opd_avx_g.o cxx/opd_avx_wrap_g.o -o _opd_avx_g.so


opd_start_fom_S: cxx/opd_xmm_ch.s
	$(CPP) -fPIC $(XMM) -c cxx/opd_xmm_ch.s -o cxx/opd_xmm.o
	$(CPP) -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so
