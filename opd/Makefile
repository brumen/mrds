# to compile xmm files
SWIG = swig
CPP  = g++-8
PROC_DIR = -funroll-all-loops -falign-functions=16 -O3
PROC_DIR_AVX = -funroll-all-loops -falign-functions=32 -O3
XMM = $(PROC_DIR) -msse4a
AVX2 = $(PROC_DIR_AVX) -mavx2  # IS THIS CORRECT? SHOULD BE PROC_DIR_AVX
PYTHON_INCLUDE = -I/usr/include/python3.5/

opd_xmm: opd_xmm.i opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i
	$(CPP) $(XMM) $(PYTHON_INCLUDE) -fPIC -c opd_xmm.c -o cxx/opd_xmm.o
	$(CPP) $(XMM) $(PYTHON_INCLUDE) -fPIC -S opd_xmm.c -o cxx/opd_xmm.s
	$(CPP) $(XMM) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
	$(CPP) -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so

# debugging version of the above
opd_xmm_debug: opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i
	$(CPP) -g $(XMM) -fPIC -c opd_xmm.c -o cxx/opd_xmm.o
	$(CPP) -g $(XMM) -fPIC -S opd_xmm.c -o cxx/opd_xmm.s
	$(CPP) -g $(XMM) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
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
	$(CPP) $(AVX2) $(PYTHON_INCLUDE) -fPIC -c opd_avx.c -o cxx/opd_avx.o
	$(CPP) $(AVX2) $(PYTHON_INCLUDE) -fPIC -S opd_avx.c -o cxx/opd_avx.s
	$(CPP) $(AVX2) $(PYTHON_INCLUDE) -fPIC -E opd_avx.c -o cxx/opd_avx.pre
	$(CPP) $(AVX2) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_avx_wrap.c -o cxx/opd_avx_wrap.o
	$(CPP) -shared  cxx/opd_avx.o cxx/opd_avx_wrap.o -o _opd_avx.so

opd_avx_test:


# debugging version of the above
opd_avx_debug: opd_avx.i opd_avx.c
	$(SWIG) -python -o cxx/opd_avx_wrap.c -I. -Wall opd_avx.i
	$(CPP) -g $(AVX2) $(PYTHON_INCLUDE) -fPIC -c opd_avx.c -o cxx/opd_avx_g.o
	$(CPP) -g $(AVX2) $(PYTHON_INCLUDE) -fPIC -S opd_avx.c -o cxx/opd_avx_g.s
	$(CPP) -g $(AVX2) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_avx_wrap.c -o cxx/opd_avx_wrap_g.o
	$(CPP) -shared  cxx/opd_avx_g.o cxx/opd_avx_wrap_g.o -o _opd_avx_g.so

opd_start_fom_S: cxx/opd_xmm_ch.s
	$(CPP) -fPIC $(XMM) -c cxx/opd_xmm_ch.s -o cxx/opd_xmm.o
	$(CPP) -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so
