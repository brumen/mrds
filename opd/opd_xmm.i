%module opd_xmm
%include <cpointer.i>

%{
  #define SWIG_FILE_WITH_INIT
  #define PY_ARRAY_UNIQUE_SYMBOL opd_xmm
  #include "opd_xmm.h"
%}

%include "opd_xmm.h"
