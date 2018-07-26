# to compile xmm files 
SWIG = swig
PROC_DIR = -funroll-all-loops -falign-functions=16 -O3
PROC_DIR_AVX = -funroll-all-loops -falign-functions=32 -O3
XMM = $(PROC_DIR) -msse4a
AVX2 = $(PROC_DIR) -mavx2
PYTHON_INCLUDE = -I/usr/include/python3.5/

opd_xmm: opd_xmm.i opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i 
	g++ $(XMM) $(PYTHON_INCLUDE) -fPIC -c opd_xmm.c -o cxx/opd_xmm.o
	g++ $(XMM) $(PYTHON_INCLUDE) -fPIC -S opd_xmm.c -o cxx/opd_xmm.s
	g++ $(XMM) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
	g++ -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so 

# debugging version of the above 
opd_xmm_debug: opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i 
	g++ -g $(XMM) -fPIC -c opd_xmm.c -o cxx/opd_xmm.o
	g++ -g $(XMM) -fPIC -S opd_xmm.c -o cxx/opd_xmm.s
	g++ -g $(XMM) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
	g++ -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so 


opd_start_fom_S: cxx/opd_xmm_ch.s
	g++ -fPIC $(XMM) -c cxx/opd_xmm_ch.s -o cxx/opd_xmm.o
	g++ -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so 

opd_1: opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i 
	g++ $(XMM) -fPIC -c cxx/opd_xmm_ch.s -o cxx/opd_xmm_ch.o
	g++ $(XMM) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
	g++ -shared cxx/opd_xmm_ch.o cxx/opd_xmm_wrap.o -o _opd_xmm.so 


# avx part
opd_avx: opd_avx.i opd_avx.c
	$(SWIG) -python -o cxx/opd_avx_wrap.c -I. -Wall opd_avx.i 
	g++ $(AVX2) $(PYTHON_INCLUDE) -fPIC -c opd_avx.c -o cxx/opd_avx.o
	g++ $(AVX2) $(PYTHON_INCLUDE) -fPIC -S opd_avx.c -o cxx/opd_avx.s
	g++ $(AVX2) $(PYTHON_INCLUDE) -fPIC -E opd_avx.c -o cxx/opd_avx.pre
	g++ $(AVX2) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_avx_wrap.c -o cxx/opd_avx_wrap.o
	g++ -shared  cxx/opd_avx.o cxx/opd_avx_wrap.o -o _opd_avx.so 

# debugging version of the above 
opd_avx_debug: opd_avx.i opd_avx.c
	$(SWIG) -python -o cxx/opd_avx_wrap.c -I. -Wall opd_avx.i 
	g++ -g $(AVX2) $(PYTHON_INCLUDE) -fPIC -c opd_avx.c -o cxx/opd_avx_g.o
	g++ -g $(AVX2) $(PYTHON_INCLUDE) -fPIC -S opd_avx.c -o cxx/opd_avx_g.s
	g++ -g $(AVX2) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_avx_wrap.c -o cxx/opd_avx_wrap_g.o
	g++ -shared  cxx/opd_avx_g.o cxx/opd_avx_wrap_g.o -o _opd_avx_g.so 

opd_start_fom_S: cxx/opd_xmm_ch.s
	g++ -fPIC $(XMM) -c cxx/opd_xmm_ch.s -o cxx/opd_xmm.o
	g++ -shared cxx/opd_xmm.o cxx/opd_xmm_wrap.o -o _opd_xmm.so 

opd_1: opd_xmm.c
	$(SWIG) -python -o cxx/opd_xmm_wrap.c -I. -Wall opd_xmm.i 
	g++ $(XMM) -fPIC -c cxx/opd_xmm_ch.s -o cxx/opd_xmm_ch.o
	g++ $(XMM) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/opd_xmm_wrap.c -o cxx/opd_xmm_wrap.o
	g++ -shared cxx/opd_xmm_ch.o cxx/opd_xmm_wrap.o -o _opd_xmm.so 
