__global__ void maximum_cuda ( float *M, int nb_cols, int nb_rows, int to_do_rows ) {

  int ind1, res_idx;
  int th_idx = threadIdx.x;
  int th_bl_idx = th_idx + blockIdx.x * blockDim.x;

  for (ind1 = 0; ind1 < (to_do_rows); ind1 = ind1 + 1) {
    res_idx = ind1 * (nb_cols) * 65535 + th_bl_idx;
    if ( res_idx < ( nb_rows ) * (nb_cols) )
      M[res_idx] = max (M[res_idx], 0.0);
  }

}

__global__ void maximum_cuda_old ( float *M, int nb_cols, int nb_rows ) {

  //int row_idx = blockIdx.y; 
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int elt_idx = blockIdx.y * nb_cols + col_idx;

  if ( col_idx < nb_cols )
      M[elt_idx] = max (M[elt_idx], 0.0);

}
