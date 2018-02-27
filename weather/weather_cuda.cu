/* vector + matrix kernel */

/* vector + matrix code */

__global__ void vpm (float *v, float *m ) {
  
  int res_idx = threadIdx.x + blockIdx.x * blockDim.x;

  m[ res_idx ] += v[blockIdx.x];
  
}
