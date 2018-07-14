// product of 9 vectors
// v is size 9*n, res is of size n 
__global__ void t1(float *v, float *res_o, int n) {
  int th_idx = blockIdx.x;
  int th_idx9 = 9 * th_idx;
  float res = v[th_idx9];
  if (th_idx < n) {
    res += v[th_idx9 + 1];
    res += v[th_idx9 + 2];
    res += v[th_idx9 + 3];
    res += v[th_idx9 + 4];
    res += v[th_idx9 + 5];
    res += v[th_idx9 + 6];
    res += v[th_idx9 + 7];
    res += v[th_idx9 + 8];
    res_o[th_idx] = res;
  }
}
