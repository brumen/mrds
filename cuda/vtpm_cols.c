// similar to vtpm, except that the multiplication/addition is done on cols
// vector + matrix slicing kernel - vpm
// vector * matrix slicing kernel - vtm
// TO CORRECT: N_STEP IS FIXED.


__global__ void vpm_cols(FLOAT_TYPE *v, FLOAT_TYPE *m, INT_TYPE nb_cols ) {

  INT_TYPE row_idx = blockIdx.y;
  INT_TYPE col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  FLOAT_TYPE v_curr = v[row_idx];

  if ( col_idx < nb_cols )
    m[row_idx * nb_cols + col_idx ] += v_curr;

}

__global__ void vtm_cols(FLOAT_TYPE *v, FLOAT_TYPE *m, INT_TYPE nb_cols) {

  INT_TYPE row_idx = blockIdx.y;
  INT_TYPE col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ FLOAT_TYPE v_curr;

  if (threadIdx.x == 0)
    v_curr = v[row_idx];

  if ( col_idx < nb_cols )
    m[row_idx * nb_cols + col_idx ] *= v_curr;

}

__global__ void vtm_cols2(FLOAT_TYPE v, FLOAT_TYPE *m, INT_TYPE nb_cols, INT_TYPE row_idx ) {

  INT_TYPE col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( col_idx < nb_cols )
    m[row_idx * nb_cols + col_idx ] *= v;

}

__global__ void vpm_cols2(FLOAT_TYPE v, FLOAT_TYPE *m, INT_TYPE nb_cols, INT_TYPE row_idx ) {

  INT_TYPE col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( col_idx < nb_cols )
    m[row_idx * nb_cols + col_idx ] += v;

}
