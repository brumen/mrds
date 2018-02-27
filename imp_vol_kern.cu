/* read from python code */
#define K_size %(K_size)d
#define ttm_size %(ttm_size)d

/* computing implied volatilities */
__device__ float comp_imp_dev (float S0, float K, float ttm, float sigma_0, \
			       float A, float B, float C, float P, float alphaC, float alphaP) {

    float z  = logf ( K / S0) / (sigma_0 * sqrtf ( ttm ) );
    float hC = z/ powf ( 1.0 + z * z, alphaC /2.0 );
    float hP = z/ powf ( 1.0 + z * z, alphaP /2.0 );
        
    return sigma_0 * sqrtf( 1.0 + A * logf ( B * expf ( C * hC) + \
					     ( 1.0 - B ) * expf (- P * hP) ) );
}

/* implied vol for a model given 
   vector of strikes K_v 
   and vectors of time-to-maturity ttm_v 
   K_v ... vector of strikes 
   ttm_v ... vector of time-to-maturity
   c ... jw7 params 
   vol_mat ... matrix where the implied vol is written 
*/
__global__ void comp_imp_vol (float *vol_mat, float *c, float *K_v, float *ttm_v) {

  int vol_idx = threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */

  if (vol_idx < K_size * ttm_size ) {

    float S0 = c[0];
    float sigma_0 = c[1];
    float A = c[2];
    float B = c[3];
    float C = c[4];
    float P = c[5];
    float alphaC = c[6];
    float alphaP = c[7];
    
    float K = K_v[threadIdx.x];
    float ttm = ttm_v [blockIdx.x];
    
    vol_mat[vol_idx] = comp_imp_dev (S0, K, ttm, sigma_0, \
				     A, B, C, P, alphaC, alphaP);
  }
  
}

/* implied vol for a model given 
   vector of strikes K_v 
   and vectors of time-to-maturity ttm_v 
   K_v ... vector of strikes 
   ttm_v ... vector of time-to-maturity
   c ... jw7 params (vector of 8x1)
   vol_mat ... matrix where the implied vol is written 
*/
__global__ void comp_vols_mat (float *vol_mat, float *S_mat, 
			       float *c, float *ttm_v) {

  int vol_idx = threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */

  if (vol_idx < K_size * ttm_size ) {

    float S0 = c[0];
    float sigma_0 = c[1];
    float A = c[2];
    float B = c[3];
    float C = c[4];
    float P = c[5];
    float alphaC = c[6];
    float alphaP = c[7];
    
    float ttm = ttm_v [blockIdx.x];
    
    vol_mat[vol_idx] = comp_imp_dev (S_mat[vol_idx], S0, ttm, sigma_0, \
				     A, B, C, P, alphaC, alphaP);
  }
  
}

/* experimenting with the above function */
__global__ void comp_vols_mat_2 (float *vol_mat, float *S_mat, 
				 float *c, float *ttm_v, 
				 float *ind_start, float *ind_stop) {

  int vol_idx = (int) (*ind_start) + threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */

  if (vol_idx < (int) (*ind_stop) ) {

    float S0 = c[0];
    float sigma_0 = c[1];
    float A = c[2];
    float B = c[3];
    float C = c[4];
    float P = c[5];
    float alphaC = c[6];
    float alphaP = c[7];
    
    float ttm = ttm_v [blockIdx.x];
    
    vol_mat[vol_idx] = comp_imp_dev (S_mat[vol_idx], S0, ttm, sigma_0, \
				     A, B, C, P, alphaC, alphaP);
  }
  
}






/* computing implied volatilities */
/* CHECK IF THIS IS TRUE, CHECK CHECK */
__global__ void comp_local_vol (float *vol_mat, float *c, float *K_v, float *ttm_v) {

  int vol_idx = threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */

  if (vol_idx < K_size * ttm_size ) {

    float S_0 = c[0];
    float sigma_0 = c[1];
    float A = c[2];
    float B = c[3];
    float C = c[4];
    float P = c[5];
    float alphaC = c[6];
    float alphaP = c[7];
    
    float K = K_v[threadIdx.x];
    float ttm = ttm_v [blockIdx.x];

    float z  = logf ( K / S_0) / (sigma_0 * sqrtf ( ttm ) );
    float sigma = comp_imp_dev( S_0, K, ttm, sigma_0,			\
				A, B, C, P, alphaC, alphaP);  /* WRONG WRONG WRONG - CHECK */
    
    float S0_local = S_0; /* CHECK CHECK CHECK CHECK  */

    float d1 = ( logf ( S0_local / S_0 ) + sigma * sigma * ttm / 2.0 ) / ( sigma * sqrtf ( ttm ) );
    float d2 = d1 - sigma * sqrtf ( ttm );
    float Xz = B * expf ( C * z) + (1.0 - B) * expf (- P * z);
        
    float sigmaK = A / (2.0 * Xz * K * sqrtf (ttm) ) / ( sqrtf ( 1.0 + A * logf (Xz) ) ) * \
      ( B * C * expf ( C * z ) - P * ( 1.0 - B ) * expf ( - P * z ) );
	
    float d1K = ( (- 1.0 / K + sigma * ttm * sigmaK ) * sigma * sqrtf ( ttm) - \
		  ( logf ( S0_local / K ) + sigma * sigma * ttm / 2.0 ) * sqrtf (ttm) * sigmaK ) / \
      ( sigma * sigma * ttm );
        
    float d2K = ( (- 1.0 / K - sigma * ttm * sigmaK ) * sigma * sqrtf ( ttm) - \
		  ( logf ( S0_local / K ) - sigma * sigma * ttm / 2.0 ) * sqrtf (ttm) * sigmaK ) / \
      ( sigma * sigma * ttm );
	
    float denomin = ( sigma_0 * sqrtf (ttm) * K * Xz  * sqrtf ( 1.0 + A * logf ( Xz ) ) );
    float BCexpr = ( B * C * expf ( C * z ) - P * (1.0 - B) * expf ( - P * z) );
        
    float sigmaKK = A / (2.0 * sqrtf (ttm)) * ( - A / ( 2.0 * denomin * K * Xz * (1.0 + A * logf (Xz) ) ) * \
                                                  BCexpr * BCexpr - BCexpr * BCexpr / ( denomin * K * Xz ) + \
						( B * C * C * expf ( C * z ) + P * P * ( 1.0 - B) * expf ( - P * z) ) / \
                                                  ( denomin * K ) - BCexpr * sigma_0 * sqrtf (ttm) / (denomin * K) \
						);
        
    float zt = logf ( K / S0_local ) / sigma_0 * ( -0.5 * powf ( ttm, - 1.5 ) ); /* derivative
										    of z wrt t  */
    float sigmat = sigma_0 * sigma_0 / (2.0 * sigma) * A / Xz  *	\
      ( B * C * expf ( C * z) - P * (1 - B) * expf ( - P * z ) ) *	\
      zt; /* derivative of sigma wrt t */
        
    float up_part = sigma * sigma + 2.0 * ttm * sigma * sigmat ;
        
    float down_part = powf ( 1.0 + K * d1 * sqrtf (ttm) * sigmaK, 2.0 ) + K * K * ttm * sigma * \
      ( sigmaKK - d1 * sigmaK * sigmaK * ttm);
        

    /* TO IMPROVE TO IMPROVE TO IMPROVE */
    vol_mat[vol_idx] = ( (up_part / down_part) < 0.0) * sigma_0 + \
      ( (up_part / down_part) >= 0.0) * sqrtf (up_part/down_part);

  }
  
}
