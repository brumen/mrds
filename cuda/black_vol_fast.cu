/* cuda fast black vol determination */
/* read from python code */
#include <math.h>

#define K_size %(K_size)d
#define ttm_size %(ttm_size)d


__device__ float normcdf (float x) {
  float L, K, w ;
  float const a1 = 0.31938153, a2 = -0.356563782, a3 = 1.781477937;
  float const a4 = -1.821255978, a5 = 1.330274429;

  L = fabs(x);
  K = 1.0 / (1.0 + 0.2316419 * L);
  w = 1.0 - 1.0 / sqrt(2 * M_PI) * exp(-L *L / 2) * (a1 * K + a2 * K *K + a3 * pow(K,3) + a4 * pow(K,4) + a5 * pow(K,5));

  if (x < 0 )
    w= 1.0 - w;
  
  return w;
}



/* computes the kirk spread option */
__device__ float spread_option_kirk (float S_1, float S_2, float K, float T, float r, 
				     float sigma_1, float sigma_2, float rho) {

  float S_A = S_1 / (S_2 + K);
  float sigma = sqrt ( sigma_1*sigma_1 + sigma_2 * sigma_2 * S_2 * S_2 / (S_2 + K) / (S_2 + K) - \
		       2 * rho * sigma_1 * sigma_2 * S_2 / (S_2 + K ) );
  
  float d_1 = (log (S_A) + 0.5 * sigma * sigma * T) / sigma / sqrt (T);

  return (S_2 + K) * exp (-r * T) * (S_A * normcdf (d_1) - normcdf (d_1 - sigma * sqrt(T) ) );
  
}



/* /\* function used in bivariate spread *\/ */
/* params ... vector as in spread_option_integral 
   z ... integrating parameter 
*/
__device__ float d11 (float *params, float z) {
  
  float alpha_1 = params[0];
  float alpha_2 = params[1]; 
  float S_1 = params[2];
  float S_2 = params[3];
  float K = params[4];
  float T = params[5];
  float r = params[6];
  float sigma_1 = params[7];
  float sigma_2 = params[8];
  float rho = params[9];
  
  /* accessory variables */
  float nu_1 = sigma_1 * sqrt(T);
  float nu_2 = sigma_2 * sqrt(T);
  float mu_1 = log(S_1) - 0.5 * sigma_1 * sigma_1 * T;
  float mu_2 = log(S_2) - 0.5 * sigma_2 * sigma_2 * T;
  float m_over = rho * z;
  float s_sqr = 1. - rho * rho;
  float s_sqrt = sqrt(s_sqr);
  float V = (log(exp(z * nu_2 + mu_2) + K) - mu_1)/nu_1;

  float cdf_int_1 = (m_over - V)/s_sqrt + s_sqrt*nu_1;
  float cdf_int_2 = cdf_int_1 - s_sqrt*nu_1;
  float H_1 = exp(mu_1 + m_over * nu_1 + 0.5 * s_sqr * nu_1* nu_1) * normcdf(cdf_int_1);
  float H_2 = (exp(z * nu_2 + mu_2) + K) * normcdf(cdf_int_2);
  
  return H_1 - H_2;
}


/* int_R N(d_11(params,z) n(z) dz = 
 1/sqrt(pi) * sum_i w_i * N(d_11(params, sqrt(2)* x_i) 
 N ... number of points in the discretization 
*/
__device__ float d11_int (float *params, int N, 
			  float *p_gh, float *w_gh) {
  
  float s = 0.; /* sum */ 
  int idx;

  for (idx=0; idx<N; idx=idx+1) 
    s = s + w_gh[idx] * d11(params, sqrt(2.) * p_gh[idx] );

  return s / sqrt(M_PI);
}

__global__ void spread_option_cva_exact (float *S_1_mat, float *S_2_mat, float T, 
					 float sigma_1, float sigma_2, float rho, 
					 float K, float r, 
					 float *t_v, 
					 float *spread_mat, 
					 float *p_gh, float *w_gh, 
					 int mat_size) {

  int vol_idx = threadIdx.x + blockIdx.x * blockDim.x;

  float S_1 = S_1_mat[vol_idx];
  float S_2 = S_2_mat[vol_idx]; 
  float ttm = T - t_v[threadIdx.x]; 
  float DF = exp (-r * ttm);
  float params[] = { 1., 1., S_1, S_2, K, ttm, r, sigma_1, sigma_2, rho };

  if (vol_idx < mat_size)
    spread_mat[vol_idx] = DF * d11_int(params, 11, p_gh, w_gh);

}








/* function used in bivariate spread, same as above  */
__device__ float d12 (float *params, float z, 
		      float *p_gh, float *w_gh) {

  float alpha_1 = params[0];
  float alpha_2 = params[1]; 
  float S_1 = params[2];
  float S_2 = params[3];
  float K = params[4];
  float T = params[5];
  float r = params[6];
  float sigma_1 = params[7];
  float sigma_2 = params[8];
  float rho = params[9];
  
  return  ( log ( alpha_1 * S_1 * exp (-r * T - rho * rho * sigma_1 * sigma_1 * T * 0.5 +
				       rho * sigma_1 * sigma_2 * sqrt (T) + rho * sigma_1 * sqrt(T) * z ) /
		  ( K - alpha_2 * S_2 * exp (- r * T - sigma_2 * sigma_2 * T * 0.5 + sigma_1 * sigma_1 * T + sigma_2 * sqrt(T) * z ) ) ) - \
	    0.5 * (1. - rho * rho ) * sigma_1 * sigma_1 * T  ) /
    ( sigma_1 * sqrt ( T * (1. - rho * rho ) ) );
}

__device__ float d12_int (float *params, int N, float *p_gh, float *w_gh) {
  
  float s = 0.; /* sum */ 
  int idx;

  for (idx=0; idx<N; idx=idx+1) 
    s = s + w_gh[idx] * d12(params, sqrt(2.) * p_gh[idx], p_gh, w_gh );
  
  return s/sqrt(M_PI);
}


/* function used in bivariate spread, same as above  */
__device__ float d2 (float *params, float z) {

  float alpha_1 = params[0];
  float alpha_2 = params[1]; 
  float S_1 = params[2];
  float S_2 = params[3];
  float K = params[4];
  float T = params[5];
  float r = params[6];
  float sigma_1 = params[7];
  float sigma_2 = params[8];
  float rho = params[9];
  
  return  ( log ( alpha_1 * S_1 * exp (-r * T - rho * rho * sigma_1 * sigma_1 * T * 0.5 +
				       rho * sigma_1 * sqrt (T) * z  + rho * sigma_1 * sigma_2 * T ) /
		  ( K - alpha_2 * S_2 * exp (- r * T - sigma_2 * sigma_2 * T + rho * sigma_1 *
					     sigma_2 * T + sigma_2 * sqrt (T) * z ) ) ) + \
	    0.5 * (1. - rho * rho ) * sigma_1 * sigma_1 * T  ) /
    ( sigma_1 * sqrt ( T * (1. - rho * rho ) ) );
}

__device__ float d2_int (float *params, int N, float *p_gh, float *w_gh) {
  
  float s = 0.; /* sum */ 
  int idx;

  for (idx=0; idx<N; idx=idx+1) 
    s = s + w_gh[idx] * d2(params, sqrt(2.) * p_gh[idx] );

  return  s / sqrt(M_PI);
}


/* 
   spread option based on the integral 
 */
__device__ float spread_option_integral (float alpha_1, float alpha_2, 
					 float S_1, float S_2, float K, float T, float r,
					 float sigma_1, float sigma_2, float rho, 
					 float *p_gh, float *w_gh) {

  float params[] = { alpha_1, alpha_2, S_1, S_2, K, T, r, sigma_1, sigma_2, rho };

  return d11_int (params, 11, p_gh, w_gh);
}








/* rows of S_1_mat are time paths, columns are simulations at the same time point 
   t_v are time points 
*/
__global__ void spread_option_cva (float *S_1_mat, float *S_2_mat, float T, 
				   float sigma_1, float sigma_2, float rho, 
				   float *t_v, 
				   float *spread_mat) {

  int vol_idx = threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */

  float S_1 = S_1_mat[vol_idx]; /* WRONG, CHECK IF threadIdx is CORRECT HERE */
  float S_2 = S_2_mat[vol_idx]; 
  // float ttm = T - t_v[blockIdx.x]; 
  
  if (vol_idx < K_size * ttm_size )
    spread_mat[vol_idx] = spread_option_kirk (S_1, S_2, 0.0, 5.0, 0.02, sigma_1, sigma_2, rho);
}

/* same as above, just with index start and stop known */ 
__global__ void spread_option_cva_2 (float *S_1_mat, float *S_2_mat, float *T, 
				     float *sigma_1_mat, float *sigma_2_mat, float rho, 
				     float *t_v, 
				     float *spread_mat, 
				     int *ind_start, int *ind_stop) {

  int vol_idx = (int) (*ind_start) + threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */

  float S_1 = S_1_mat[vol_idx]; /* WRONG, CHECK IF threadIdx is CORRECT HERE */
  float S_2 = S_2_mat[vol_idx]; 
  // float ttm = T - t_v[blockIdx.x]; 
  float sigma_1 = sigma_1_mat[vol_idx]; 
  float sigma_2 = sigma_2_mat[vol_idx]; 
  
  if (vol_idx < (int) (*ind_stop) )
    spread_mat[vol_idx] = spread_option_kirk (S_1, S_2, 0.0, 5.0, 0.02, sigma_1, sigma_2, rho);

}








/* black-scholes for volatility 
   computes black-scholes prices for prices 
   vol_mat ... volatility matrix 
   bs_price_mat ... black_scholes price matrix 
   S0_mat ... matrix of S0-s
   K .. strike 
   ttm_v ... vector of times to maturity 
 */
__global__ void comp_black_scholes (float *bs_price_mat, 
				    float *vol_mat, float *S0_mat, float *K, float *ttm_v) {

  int vol_idx = threadIdx.x + blockIdx.x * blockDim.x ; /* block index, goes over simulations */

  float S0 = S0_mat[threadIdx.x]; /* WRONG, CHECK IF threadIdx is CORRECT HERE */
  float ttm = ttm_v[blockIdx.x]; /* AS ABOVE */
  
  float d1 = ( log (S0 / *K ) + 0.5 * vol_mat[vol_idx] * vol_mat[vol_idx] * ttm) / \
    (vol_mat[vol_idx] * sqrt (ttm) );


  if (vol_idx < K_size * ttm_size ) {
    
    /* 
       float K = K_v[threadIdx.x];
       float ttm = ttm_v [blockIdx.x];
    */
    bs_price_mat[vol_idx] = S0 * normcdf(d1) - (*K) * normcdf(d1 - vol_mat[vol_idx] * sqrt(ttm) );
    
    /* 
    vol_mat[vol_idx] = comp_imp_dev (S0, K, ttm, sigma_0,		\
				     A, B, C, P, alphaC, alphaP);
    */
  }
  
}






/* inverse cumulative normal distribution */
__device__ float normsinv (float p) { 
  /* inverse cumulative normal distribution by jacklam@math.uio.no: 
     http://cbio.mskcc.org/~jansen/normsinv/source/normsinv.cpp
  */

  //const float HUGE_VAL = 100.;
  const float ERANGE = 1.0;
  const float EDOM = -1.0;

  const float LOW = 0.02425; 
  const float HIGH = 0.97575; // Coefficients in rational approximations. 
  const float a[] = { -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 }; 
  const float b[] = { -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 }; 
  const float c[] = { -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 }; 
  const float d[] = { 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 }; 

  float q, r; 
  int errno1 = 0; 

  if (p < 0 || p > 1) { errno1 = EDOM; return 0.0; } 
  else if (p == 0) { errno1 = ERANGE; return -HUGE_VAL; } 
  else if (p == 1) { errno1 = ERANGE; return HUGE_VAL; } 
  else if (p < LOW) { 
    q = sqrt(-2*log(p)); 
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); 
  } else if (p > HIGH) { 
    q = sqrt(-2*log(1-p)); 
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); 
  } else { 
    q = p - 0.5; r = q*q; 
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1); 
  } 
} 



/* helper function for black_vol... function below 
   implements normalized black price 
   theta ... 1 for call, - WRONG WRONG WRONG - CHECK HERE. 
*/
__device__ float b (float x, double sigma, float theta) {
  float e1 = exp (x/2);
  float d1 = x/sigma + sigma/2. ;
  return theta * e1 * normcdf (theta * d1 ) -		\
    theta /e1 * normcdf (theta * (d1 - sigma) );
}



__device__ double black_vol_inverse_normalized_fast (double beta, double x, int theta, double tol) {


  float sigma_c = sqrt (2. * abs(x) ); /* inflection point function  */
  float iota = (theta*x > 0) * theta * (exp (x/2) - exp (-x/2)); 
  float b_c = b (x, sigma_c, 1. ); /* b (sigma_c) */
  float sigma_low = sqrt ( (2.0 * x*x )/ ( abs(x) - 4.0 * log ( (beta - iota)/( b_c - iota ) ) ) );
  float e1 = exp ( theta * x / 2.0 );
  float sigma_high = -2.0 * normsinv (  (e1 - beta ) / (e1 - b_c ) *	\
					normcdf ( - sqrt ( abs(x) / 2.0) ) );

  /* iteration */
  float sigma; 
  if (beta < b_c)  /* initial value of sigma */
    sigma = sigma_low;
  else /* (beta >= b_c) */
    sigma = sigma_high;
	  
  float sigma_new = sigma * (1.0 + 2.0 * tol); /* initial value of
						  changed sigma, such that delta_sigma / sigma > tol; */
  float delta_sigma = 2.0 * tol; /* this is actually precise */

  float b_der;
  float one_step; 

  while ( delta_sigma / sigma > tol ) {
    b_der = exp ( - 0.5 * pow( x/sigma, 2) - 0.5 * pow(sigma/2., 2) ) / sqrt (2. * 3.14);
    if (beta < b_c )
      one_step = log ( (beta - iota) / (b(x, sigma, 1) - iota) ) * (b(x, sigma, 1) - iota ) / b_der;
    else
      one_step = (beta - b(x, sigma, 1))/b_der ;

    sigma_new = sigma + one_step;
    delta_sigma = abs (sigma_new - sigma);
    sigma = sigma_new;
  }
  
  return sigma_new;

}


