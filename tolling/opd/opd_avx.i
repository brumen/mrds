%module opd_avx
%include <cpointer.i>

%{
  #define SWIG_FILE_WITH_INIT
  #define PY_ARRAY_UNIQUE_SYMBOL opd_avx
  #include "opd_avx.h"
%}

%include "opd_avx.h"
