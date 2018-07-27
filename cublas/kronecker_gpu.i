%module kronecker_gpu

%{
  #include "/usr/include/cuda.h"
  //#include "/usr/local/cuda/include/cuda_runtime_api.h"
  //#include "/usr/local/cuda/include/curand.h"
  //#include </usr/local/cuda/include/thrust/device_vector.h>
  #include "kronecker_gpu.h"
  //void vpv(float *v1_p, float *v2_p, float *m_new_p, 
  //  	   int v1_size, int v2_size);

%}

//%include "/usr/local/cuda/include/cuda.h"
%include "kronecker_gpu.h"

