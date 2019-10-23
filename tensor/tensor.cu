/* read from python code */
/* used in tolling.py code */
#define lattice_size %(lss)d



/*
  computes P_m * G_m + H_m  (where P_m tensor, G_m, H_m matrices)
*/
__global__ void tensor_P_m (float *P_m, float *G_m, float *H_m, float *res_m ) {

  int res_idx = threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */
  int ind1, ind2;
  int nb_r = lattice_size;
  int nb_c = lattice_size;

  res_m[res_idx] = 0.;

  for (ind1=0; ind1< nb_r ; ind1 = ind1 + 1)
    for (ind2=0; ind2 < nb_c; ind2 = ind2 + 1)
      res_m[res_idx] +=  P_m[res_idx* nb_r * nb_c + ind1 * nb_c + ind2 ] * G_m[ind1 * nb_c + ind2];
  res_m[res_idx] += H_m[res_idx];
}

/*
  computes P_m * G_m  (where P_m tensor, G_m matrix), simpler version of above
*/
__global__ void tensor_P_m_simple (float *P_m, float *G_m, float *res_m ) {

  int res_idx = threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */
  int ind1, ind2;
  int nb_r = lattice_size;
  int nb_c = lattice_size;

  res_m[res_idx] = 0.0; /* initial value */
  for (ind1=0; ind1< nb_r ; ind1 = ind1 + 1)
    for (ind2=0; ind2 < nb_c; ind2 = ind2 + 1)
      res_m[res_idx] += P_m[res_idx* nb_r * nb_c + ind1 * nb_c + ind2 ] * G_m[ind1 * nb_c + ind2];
}


/*
  Alternative version of tensor prod, faster than above
  computes P_m * G_m + H_m  (where P_m tensor, G_m, H_m matrices)
*/
__global__ void tensor_P_m_alt (float *P_m, float *G_m, float *H_m, float *res_m ) {

  __shared__ float P_m_shared[32][32];
  __shared__ float G_m_shared[32][32];
  __shared__ float res_m_shared[32][32];
  int block_idx = blockIdx.x * lattice_size + blockIdx.y ; /* block index */
  int thread_idx = threadIdx.x;

  int lat_sq = lattice_size * lattice_size;

  /* Below %% is used for python processing */
  int End = lattice_size %% 32 == 0 ? lattice_size / 32 : (lattice_size/32) + 1; /* tiling */
  int i, j, k;

  /* for (i=0; i<End; i++) { */
  /* i = blockIdx.x */

  for (j=0; j<End; j++) {
    for (k=0; k<32; k++) {
      cnd_1 = block_idx * lat_sq + blockIdx.x * 32 * lattice_size + k * lattice_size + j * 32 + thread_idx < lat_sq * lat_sq;
      cnd_2 = blockIdx.x * 32 * lattice_size + k * lattice_size + j * 32 + thread_idx < lat_sq * lattice_size;
      cnd_3 = blockIdx.x * 32 *lattice_size + k * lattice_size + j * 32 + thread_idx < lat_sq;
      cnd_4 = j * 32 + thread_idx < lattice_size;

      if ( cnd_1 && cnd_2 && cnd_3 && cnd_4 ) {
        P_m_shared[k][thread_idx] = P_m[ block_idx * lat_sq + blockIdx.x * 32 * lattice_size + k * lattice_size + j * 32 + thread_idx];
        G_m_shared[k][thread_idx] = G_m[ blockIdx.x * 32 * lattice_size + k * lattice_size + j * 32 + thread_idx];
      }
    }
  }

  __syncthreads();

  /* tensor multiplication */
  for (i=0; i<32; i++)
    res_m_shared[thread_idx][i] = P_m_shared[thread_idx][i] * G_m_shared[thread_idx][i];

  __syncthreads();

  res_m[block_idx] = 0.0; /* initial value */

  /* reduction step */
  for (k=0; k<32; k++)
    res_m[block_idx] += res_m_shared[thread_idx][k];

  res_m[block_idx] += H_m[block_idx];

}


/* second try on the function above */
__global__ void tensor_P_m_alt2 (float *P_m, float *G_m, float *H_m, float *res_m ) {

  __shared__ float P_m_shared[32][32];
  __shared__ float G_m_shared[32][32];
  __shared__ float res_m_pr[32]; /* partially reduced result */
  int block_idx = blockIdx.x * lattice_size + blockIdx.y ; /* block index */
  int thread_idx = threadIdx.x;

  int lat_sq = lattice_size * lattice_size;

  /* %% below is used for python processing */
  int End = lattice_size %% 32 == 0 ? lattice_size / 32 : (lattice_size/32) + 1; /* tiling */ 
  int i, j, k; 
  float sum;
  int pos; /* (partial) position in the matrix */

  for (i=0; i<End; i++) { 

    for (j=0; j<End; j++) {
      pos = i * 32 * lattice_size + j * 32 + thread_idx;

      res_m_pr[thread_idx] = 0.;
      for (k=0; (k<32)
             && ( j * 32 + thread_idx < lattice_size ) /* horizontal condition */
             && ( i * 32 + k < lattice_size ) /* vertical condition */
             ; k++) {
        /* P_m_shared[k][thread_idx] = P_m[ block_idx * lat_sq + pos + k * lattice_size ];
           G_m_shared[k][thread_idx] = G_m[ pos + k * lattice_size ];
        */
        res_m_pr[thread_idx] += P_m[ block_idx * lat_sq + pos + k * lattice_size ] * G_m[ pos + k * lattice_size ];
      }

      /* partial reduction */

      /* for (k=0; ( k<32 ) */
      /* 	     && ( j * 32 + thread_idx < lattice_size ) /\* horizontal condition *\/ */
      /* 	     && ( i * 32 + k < lattice_size ) /\* vertical condition *\/ */
      /* 	     ; k++) */
      /* 	res_m_pr[thread_idx] += P_m_shared[k][thread_idx] * G_m_shared[k][thread_idx]; */

      __syncthreads();

      if (threadIdx.x == 0) { /* just the first thread sums the elts */
      	sum = 0.;
      	for (k=0; (k<32) && ( j * 32 + k < lattice_size ); k++) /* horizontal condtion */
      	  sum += res_m_pr[k];
      	res_m[block_idx] += sum;
      }

      /* reduction step, this is how it is supposed to be  */
      /* for (k=0; k<32; k++)
         atomicAdd( &sum2, sum); */

      /* MISSING MISSING MISSING */
      /* res_m[block_idx] += H_m[block_idx]; */
    }
  }
}




/*
  tensor version of cumsum function
  LOTS OF IMPROVEMENT NECESSARY, SHARED MEMORY NEEDS TO BE USED
*/
__global__ void tensor_cumsum(float *dest, float *src, int stride) {

  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  float tmp = 0.0;
  int n;

  for(n = 0; n < stride; n++) {
    tmp += src[ n + stride * i ];
    dest[ n + stride * i ] = tmp;
  }

}
