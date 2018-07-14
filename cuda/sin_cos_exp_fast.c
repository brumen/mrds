__global__ void sin_fast(float *x, float *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = sinf(x[idx]);
}

__global__ void cos_fast(float *x, float *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = cosf(x[idx]);
}
__global__ void exp_fast(float *x, float *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = expf(x[idx]);
}
