__global__ void sin_fast(double *x, double *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = sin(x[idx]);
}

__global__ void cos_fast(double *x, double *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = cos(x[idx]);
}
__global__ void exp_fast(double *x, double *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = exp(x[idx]);
}
