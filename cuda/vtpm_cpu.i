%module vtpm_cpu
%include <cpointer.i>

%{
  #define SWIG_FILE_WITH_INIT
  #define PY_ARRAY_UNIQUE_SYMBOL opd_avx
  #include "vtpm_cpu.h"
%}

%include "vtpm_cpu.h"
