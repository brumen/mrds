// vector + matrix slicing kernel - vpm
// vector * matrix slicing kernel - vtm
// TO CORRECT: N_STEP IS FIXED.

// FLOAT_TYPE: float or double,
// INT_TYPE: probably int, check if other ints are possible

__global__ void vpm(FLOAT_TYPE *v, FLOAT_TYPE *m, INT_TYPE nb_cols, INT_TYPE nb_rows, INT_TYPE to_do_rows) {
  INT_TYPE ind1, res_idx;
  INT_TYPE th_idx = threadIdx.x;
  INT_TYPE th_bl_idx = th_idx + blockIdx.x * blockDim.x;
  __shared__ FLOAT_TYPE v_cache[31];
  v_cache[th_idx] = v[th_idx];

  for (ind1 = 0; ind1 < to_do_rows; ind1 = ind1 + 1) {
    res_idx = ind1 * nb_cols * 65535 + th_bl_idx;
    if (res_idx < nb_rows * nb_cols)
      m[res_idx] += v_cache[th_idx];
  }
}

__global__ void vtm(FLOAT_TYPE *v, FLOAT_TYPE *m, INT_TYPE nb_cols, INT_TYPE nb_rows, INT_TYPE to_do_rows) {
  INT_TYPE ind1, res_idx;
  INT_TYPE th_idx = threadIdx.x;
  INT_TYPE th_bl_idx = th_idx + blockIdx.x * blockDim.x;
  __shared__ FLOAT_TYPE v_cache[31];
  v_cache[th_idx] = v[th_idx];

  for (ind1 = 0; ind1 < (to_do_rows); ind1 += 1) {
    res_idx = ind1 * (nb_cols) * 65535 + th_bl_idx;
    if ( res_idx < (nb_rows) * (nb_cols) )
      m[res_idx] *= v_cache[th_idx];
  }
}
