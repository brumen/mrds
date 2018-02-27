# call_put_ind = 0 for CALL
#                1 for PUT
def apo_long (F_curve, sigma_curve, rho_mat,	\
	      T_i_curve, t_i_curve, \
	      t, beta, sigma_L, call_put_ind):

    N = len (F_curve)
    M_1 = average (F_curve)

  int i;

  for (i=0; i<N; i++) {
    M_2_term_1[i] = F_curve[i]* F_curve[i] * exp (A(T_i_curve[i], T_i_curve[i], t, T_i_curve[i]) \
						  * sigma_curve[i] * (t_i_curve[i] - t) );
    for (j=i+1; j++; j < N) 
      M_2_matrix_term[i][j] = 2 * F_curve[i] * F_curve[j] * \
	exp ( A( T_i_curve[i], T_i_curve[j], t, min (T_i_curve[i], T_i_curve[j]) ) * \
	      rho_mat[i][j] * sigma_curve[i] * sigma_curve[j] * (t_i_curve[i] - t) );
  }
    
  float M_2 = (sum(M_2_term1) + sum(M_2_matrix_term) ) / ( (float) (N**2) );

  /* THIS BELOW HAS TO BE REWRITTEN */
  return black_greeks (M_1, K, r, sigma, T, call_put_ind)
