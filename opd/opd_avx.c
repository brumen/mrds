#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL opd_xmm
#define AVX2

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef AVX2
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif

#define PO PyObject 

#include <python2.7/Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include "opd_avx.h"


#ifdef XMM
#define DOUBLE_INCR 2
#define SHORT_INCR 8
#define reg __m128d
#define reg_int __m128i
#define madd _mm_add_pd
#define mmul _mm_mul_pd
#define mset _mm_set_pd1
#define mseti _mm_set1_epi16  // setting of integers 
#define msetz _mm_setzero_pd
#define mloa _mm_load_pd
#define msto _mm_store_pd
#define msub _mm_sub_pd
#endif

#ifdef AVX2
#define DOUBLE_INCR 4
#define SHORT_INCR 16
#define reg __m256d
#define reg_int __m256i
#define madd _mm256_add_pd
#define mmul _mm256_mul_pd
#define mset _mm256_set1_pd  // setting of doubles 
#define mseti _mm256_set1_epi16  // setting of integers 
#define msetz _mm256_setzero_pd
#define mloa _mm256_loadu_pd
#define mloa_int _mm256
#define msto _mm256_storeu_pd
#define msub _mm256_sub_pd
#endif

#define cp(name) PyArrayObject *npy_## name = (PyArrayObject *) (name)  // change pointer
#define cn(name) npy_## name->data  // change name 

void add4_internal(double *r, double *a, double *b, double *c, double *d, double *y, 
		   int nSize) {
  // computes r - (a+b+c+d)
  // fast, not much faster than add4_simple, but nSize has to be divisible by 4
  size_t idx = 0;
  size_t idx2;
  reg a_xmm, b_xmm, c_xmm, d_xmm, r_xmm;
  size_t nb_repeats = nSize/DOUBLE_INCR -1;
  
  for (idx2=0; idx2<nb_repeats; idx2 += 1) {
    a_xmm = mloa(a + idx);
    b_xmm = mloa(b + idx);
    c_xmm = mloa(c + idx);
    d_xmm = mloa(d + idx);
    r_xmm = mloa(r + idx);
    a_xmm = madd(a_xmm, b_xmm);
    c_xmm = madd(c_xmm, d_xmm);
    a_xmm = madd(a_xmm, c_xmm);
    a_xmm = msub(r_xmm, a_xmm);
    msto(y+idx, a_xmm);
    idx += DOUBLE_INCR;
  }
  // handling the last few items
  for (idx2=idx; idx2<nSize; idx2+=1) {
    y[idx2] = r[idx2] - (a[idx2] + b[idx2] + c[idx2] + d[idx2]);
  }
}

void add4_simple(double *r, double *a, double *b, double *c, double *d, double *y, 
		 int nSize) {
  // computes r - (a+b+c+d)
  // this routine is a bit slower than add4_internal, but it handles any number of elements
  size_t idx;
  for (idx=0; idx<nSize; idx += 1)
    y[idx] = r[idx] - (a[idx] + b[idx] + c[idx] + d[idx]);
}


void add4(PO *r, PO *a, PO *b, PO *c, PO *d, PO *y,
	  int n) {
  cp(r); cp(a); cp(b); cp(c); cp(d); cp(y);
  add4_internal((double *) cn(r),
		(double *) cn(a), 
		(double *) cn(b),
		(double *) cn(c), 
		(double *) cn(d),
		(double *) cn(y),
		n);
}


void mul4_internal(double *r, double *a, double *b, double c, double *y, 
		   int nSize) {
  // computes: r * a * b * c (c scalar, all other vectors)
  size_t idx;
  reg r_xmm, a_xmm, b_xmm;
  reg c_xmm = mset(c);
  for (idx=0; idx<nSize; idx += DOUBLE_INCR) {
    a_xmm = mloa(a + idx);
    b_xmm = mloa(b + idx);
    r_xmm = mloa(r + idx);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, b_xmm);
    a_xmm = madd(a_xmm, c_xmm);
    msto(y+idx, a_xmm);
  }
}

void mul4(PO *r, PO *a, PO *b, double c, PO *y,
	  int n) {
  // computes: r * a * b * c (c scalar, all other vectors)
  cp(r); cp(a); cp(b); cp(y);
  mul4_internal((double *) cn(r),
		(double *) cn(a), 
		(double *) cn(b),
		c,
		(double *) cn(y),
		n);
}

void mul5_internal(double *r, double *a, double *b, double c, double *y, 
		   int nSize) {
  // computes: r * a * b * c (c scalar, all other vectors)
  size_t idx;
  reg r_xmm, a_xmm, b_xmm;
  reg c_xmm = mset(c);
  for (idx=0; idx<nSize; idx += DOUBLE_INCR) {
    a_xmm = mloa(a + idx);
    b_xmm = mloa(b + idx);
    r_xmm = mloa(r + idx);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, b_xmm);
    a_xmm = madd(a_xmm, c_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    a_xmm = madd(a_xmm, r_xmm);
    msto(y+idx, a_xmm);
  }
}

void mul5(PO *r, PO *a, PO *b, double c, PO *y,
	  int n) {
  cp(r); cp(a); cp(b); cp(y);
  mul5_internal((double *) cn(r),
		(double *) cn(a),
		(double *) cn(b),
		c,
		(double *) cn(y),
		n);
}

void mul6_internal(double *r, double *a, double *b, double c, double *y, 
		   int nSize) {
  // computes: r * a * b * c (c scalar, all other vectors)
  size_t idx;
  reg a_xmm;
  reg c_xmm = mset(c);
  for (idx=0; idx<nSize; idx += DOUBLE_INCR) {
    a_xmm = madd(a_xmm, c_xmm);
    a_xmm = madd(a_xmm, c_xmm);
  }
}

void mul6(PO *r, PO *a, PO *b, double c, PO *y,
	  int n) {
  cp(r); cp(a); cp(b); cp(y);
  mul6_internal((double *) cn(r),
		(double *) cn(a),
		(double *) cn(b),
		c,
		(double *) cn(y),
		n);
}

void mul7_internal(double *r, double *a, double *b, double c, double *y, 
		   int nSize) {
  size_t idx;
  reg a_xmm;
  reg c_xmm = mset(c);
  reg b_xmm = mset(c+1);
  reg d_xmm = mset(c+2);
  reg e_xmm = mset(c+3);
  for (idx=0; idx<nSize; idx += DOUBLE_INCR) {
    a_xmm = mloa(a + idx);
    reg res_xmm = madd(mmul(madd(a_xmm, b_xmm),c_xmm),d_xmm);
    msto(y+idx, res_xmm);
  }
}

void mul7(PO *r, PO *a, PO *b, double c, PO *y,
	  int n) {
  cp(r); cp(a); cp(b); cp(y); 
  mul7_internal((double *) cn(r),
		(double *) cn(a),
		(double *) cn(b),
		c,
		(double *) cn(y),
		n);
}

void skew_fom_internal_avx(double F, double *delta_X, 
			   double c1_ch, double qv, 
			   double c2_ch, double c3_ch, 
			   double *sim_fom, 
			   int nb_sim) {
  size_t idx;
  reg F_xmm = mset(F);
  reg one_xmm = mset(1.);
  reg qv_xmm = mset(qv);
  reg qv_xmm2 = mmul(qv_xmm, qv_xmm);
  reg c1_ch_xmm = mset(c1_ch);
  reg c2_ch_xmm = mset(c2_ch);
  reg c3_ch_xmm = mset(c3_ch);
  reg fact_xmm = mset(-3.);

  for (idx=0; idx< nb_sim; idx += DOUBLE_INCR) {
    reg delta_X_xmm = mloa(delta_X + idx);
    reg delta_X2_xmm = mmul(delta_X_xmm, delta_X_xmm);
    reg delta_X3_xmm = mmul(delta_X2_xmm, delta_X_xmm);
    reg delta_X4_xmm = mmul(delta_X3_xmm, delta_X_xmm);
    reg ft_xmm = msub(delta_X2_xmm, qv_xmm);
    // first term
    ft_xmm = mmul(c1_ch_xmm, ft_xmm);
    // second term
    reg st_xmm = mset(-3.);
    st_xmm = mmul(st_xmm, delta_X_xmm);
    st_xmm = mmul(st_xmm, qv_xmm); // -3 * X * qv
    st_xmm = madd(delta_X3_xmm, st_xmm); // X**3 - 3 X qv
    st_xmm = mmul(c2_ch_xmm, st_xmm);  // c2 * (X**3 - 3 X qv)
    //third term
    reg tt_xmm = mset(3.);
    tt_xmm = mmul(tt_xmm, qv_xmm2);  // 3*qv*2
    // delta_X3 overwritten
    delta_X3_xmm = mset(-6.);
    delta_X3_xmm = mmul(delta_X3_xmm, qv_xmm);
    delta_X3_xmm = mmul(delta_X3_xmm, delta_X2_xmm);
    tt_xmm = madd(tt_xmm, delta_X3_xmm);
    tt_xmm = madd(tt_xmm, delta_X4_xmm);
    tt_xmm = mmul(tt_xmm, c3_ch_xmm);
    // result
    reg res_xmm = madd(one_xmm, delta_X_xmm);
    res_xmm = madd(res_xmm, ft_xmm);
    res_xmm = madd(res_xmm, st_xmm);
    res_xmm = madd(res_xmm, tt_xmm);
    res_xmm = mmul(res_xmm, F_xmm);
    msto(sim_fom+idx, res_xmm);
  }
}

void skew_fom_internal_simple(double F, double *delta_X, 
			      double c1_ch, double qv, 
			      double c2_ch, double c3_ch, 
			      double *sim_fom, 
			      int nb_sim) {
  size_t idx;
  for (idx=0; idx< nb_sim; idx += 1) {
    double delta_X_xmm = delta_X[idx];
    double delta_X2_xmm = delta_X_xmm * delta_X_xmm;
    double delta_X3_xmm = delta_X2_xmm * delta_X_xmm;
    double delta_X4_xmm = delta_X3_xmm * delta_X_xmm;
    // first term 
    double ft_xmm = c1_ch * delta_X2_xmm - qv;
    // second term
    double st_xmm = c2_ch * (delta_X3_xmm -3. * delta_X_xmm * qv);
    //third term 
    double tt_xmm = c3_ch * (delta_X4_xmm -6. * qv * delta_X2_xmm + 3. * qv*qv);
    // result
    double res_xmm = F * (1. + delta_X_xmm + ft_xmm + st_xmm + tt_xmm);
    sim_fom[idx] = res_xmm;
  }
}


void skew_fom(double F, PO *delta_X, 
	      double c1_ch, double qv, double c2_ch, double c3_ch, 
	      PO *sim_fom, 
	      int nb_sim) {
  cp(delta_X);
  cp(sim_fom);
  skew_fom_internal_simple(F, (double *) cn(delta_X),
			   c1_ch, qv, c2_ch, c3_ch, 
			   (double *) cn(sim_fom),
			   nb_sim);
}

void skew_fom_avx(double F, PO *delta_X, 
		  double c1_ch, double qv, double c2_ch, double c3_ch, 
		  PO *sim_fom, 
		  int nb_sim) {
  cp(delta_X);
  cp(sim_fom);
  skew_fom_internal_avx(F, (double *) cn(delta_X),
			c1_ch, qv, c2_ch, c3_ch, 
			(double *) cn(sim_fom),
			nb_sim);
}

// performs np.sum(vec1 * vec2), a scalar product
double num_quad_internal(double *vec1, double *vec2, size_t v_len) {
  reg v1_xmm, v2_xmm, v3_xmm, res_xmm;
  size_t idx;
  double res[DOUBLE_INCR];
  res_xmm = msetz();
  for (idx=0; idx < v_len; idx += DOUBLE_INCR) {
    v1_xmm = mloa(vec1 + idx);
    v2_xmm = mloa(vec2 + idx);
    v2_xmm = mmul(v1_xmm, v2_xmm);  // multiplied 
    res_xmm = madd(res_xmm, v2_xmm); // added 
  }
  msto(res, res_xmm);
  return res[0] + res[1];
}

double num_quad(PO *vec1, PO *vec2, int v_len) {
  cp(vec1); cp(vec2);
  return num_quad_internal((double *) cn(vec1),
			   (double *) cn(vec2),
			   (size_t) v_len);
}

void test_1_int(double *v1, double *v2, double *res, int len) {
  size_t idx;
  for (idx=0; idx<len; idx+= 1)
    res[idx] = v1[idx] + v2[idx];
}

void test_1(PO *v1, PO *v2, PO *res, int len) {
  cp(v1); cp(v2); cp(res);
  return test_1_int((double *) cn(v1),
		    (double *) cn(v2),
		    (double *) cn(res),
		    len);
}


void do_start_shut_internal(short *dc_can,
			    short *dc_force,
			    short *is_profitable,
			    short *do_action,
			    int nb) {
  // implements startup/shutdown
  // dc_can_start & ((dc_force_start == 2) || (is_profit & dc_force == 1))

  size_t idx;
  reg_int *dc_can_avx = (reg_int *) dc_can;
  reg_int *dc_force_avx = (reg_int *) dc_force;
  reg_int *is_profitable_avx = (reg_int *) is_profitable;
  reg_int *do_action_avx = (reg_int *) do_action;
  for (idx=0; idx < nb/SHORT_INCR; idx += 1) {
    reg_int can_avx = _mm256_loadu_si256(dc_can_avx + idx);
    reg_int force_avx = _mm256_loadu_si256(dc_force_avx + idx);
    reg_int is_prof_avx = _mm256_loadu_si256(is_profitable_avx + idx);
    reg_int cnd_1 = _mm256_cmpeq_epi16(force_avx, _mm256_set1_epi16(2));
    reg_int cnd_2 = _mm256_and_si256(is_prof_avx,
				     _mm256_cmpeq_epi16(force_avx, _mm256_set1_epi16(1)));

    cnd_1 = _mm256_or_si256(cnd_1, cnd_2);
    cnd_1 = _mm256_and_si256(can_avx, cnd_1);
    _mm256_storeu_si256(do_action_avx + idx, cnd_1);
  }
}


void do_start_shut_internal_simple(short *dc_can,
				   short *dc_force,
				   short *is_profitable,
				   short *do_action,
				   int nb) {
  // implements startup/shutdown, in a naive fashion 
  // dc_can_start & ((dc_force_start == 2) || (is_profit & dc_force == 1))
  // this works _MUCH_ slower than _internal_ function above
  
  size_t idx;
  for (idx=0; idx < nb; idx += 1) 
    do_action[idx] = dc_can[idx] &
      ((dc_force[idx] == 2) || (is_profitable[idx] & dc_force[idx] == 1));
}


void do_start_shut(PO *dc_can, PO *dc_force, PO *is_profitable, PO *do_action, 
		   int nb_sim) {
  cp(dc_can); cp(dc_force);
  cp(is_profitable); cp(do_action);
  do_start_shut_internal((short *) cn(dc_can),
  			 (short *) cn(dc_force),
  			 (short *) cn(is_profitable),
  			 (short *) cn(do_action),
  			 nb_sim);
}


// cold startup (storing integers, etc)
void cold_start_internal(short *hours_shut,
			 short *res,
			 short xud_cold_start,
			 int nSize) {
  size_t idx;
  reg_int cold_start_avx = mseti(xud_cold_start);
  reg_int *hours_shut_avx = (reg_int *) hours_shut;
  reg_int *res_avx  = (reg_int *) res;
  for (idx=0; idx < nSize / SHORT_INCR; idx +=1) {
    reg_int hs_avx = _mm256_load_si256(hours_shut_avx + idx);
    reg_int cnd_1 = _mm256_cmpgt_epi16(hs_avx, cold_start_avx);
    _mm256_store_si256(res_avx+idx, cnd_1);
  }
  // Missing if nSize not divisible by 16 ????
}


void cold_start(PO *hours_shut,
		PO *res,
		int xud_cold_start,
		size_t nSize) {
  cp(hours_shut); cp(res);
  cold_start_internal((short *) cn(hours_shut), 
		      (short *) cn(res),
		      (short) xud_cold_start,
		      nSize);
}

// selecting from 2 doubles according to int16
void sel_double2_internal(double *x1, double *x2,
			  short *sel, double *res,
			  int n) {
  // if s == 0xFFFF s1, else s2
  size_t idx;
  reg res_avx;
  for (idx=0; idx<n; idx+= 2) {
    reg sel_1_avx = (reg) mseti(sel[idx]);  
    reg sel_2_avx = (reg) mseti(sel[idx+1]);
    reg sel_avx  = _mm256_shuffle_pd(sel_1_avx, sel_2_avx, _MM_SHUFFLE2(0,0));
    reg x1_avx = mloa(x1+idx);
    res_avx = _mm256_xor_pd(mloa(x2+idx), x1_avx);
    res_avx = _mm256_and_pd(sel_avx, res_avx);
    res_avx = _mm256_xor_pd(x1_avx, res_avx);
    msto(res+idx, res_avx);
  }
}


void is_start_profitable_internal(double *startup_sp_in,
				  double shutdown_sp_in,
				  double *fixed_and_fuel_startup_cost,
				  double xud_startup_horizon,
				  double xud_shutdown_horizon,
				  double max_cap,
				  double *shutdown_gen_profit,
				  double *pp_marg_max,
				  short *is_shutdown_profitable,
				  short *is_startup_profitable,
				  int nSize) {

  size_t idx;
  reg xud_start_avx = mset(xud_startup_horizon);
  reg xud_shut_avx = mset(xud_shutdown_horizon);
  reg max_cap_avx = mset(max_cap);
  reg sh_max_cap_avx = mset(1./(xud_startup_horizon * max_cap));
  reg shut_sp_in_avx = mset(shutdown_sp_in);
  for (idx=0; idx < nSize; idx += 2) {
    reg start_sp_in_avx = mloa(startup_sp_in + idx);
    reg ff_startup_avx = mloa(fixed_and_fuel_startup_cost + idx);
    reg shut_gen_profit_avx = mloa(shutdown_gen_profit + idx);
    reg pp_marg_max_avx = mloa(pp_marg_max + idx);
    reg xud_startup_sp_avx = mmul(start_sp_in_avx, sh_max_cap_avx);
    xud_startup_sp_avx = madd(start_sp_in_avx, xud_startup_sp_avx);
    reg shut_cost_sp_avx = mmul(shut_sp_in_avx, mmul(xud_shut_avx, max_cap_avx));
    // double masks
    reg is_shut_prof = _mm256_cmp_pd(madd(madd(ff_startup_avx, shut_cost_sp_avx),
					    shut_gen_profit_avx),
				     msetz(), 1);  // 1 = LT
    reg is_start_prof = _mm256_cmp_pd(xud_startup_sp_avx, pp_marg_max_avx, 1);
    //saving masks
    reg_int start_mask_i = (reg_int) is_start_prof;
    reg_int shut_mask_i = (reg_int) is_shut_prof;
    is_shutdown_profitable[idx] = _mm256_extract_epi16(shut_mask_i, 3);
    is_shutdown_profitable[idx+1] = _mm256_extract_epi16(shut_mask_i, 7);
    is_startup_profitable[idx] = _mm256_extract_epi16(start_mask_i, 3);
    is_startup_profitable[idx+1] = _mm256_extract_epi16(start_mask_i, 7);
  }
  // mising if nSize not divisible by 2 
}


void is_start_profitable(PO *startup_sp_in,
			 double shutdown_sp_in,
			 PO *fixed_and_fuel_startup_cost,
			 double xud_startup_horizon,
			 double xud_shutdown_horizon,
			 double max_cap,
			 PO *shutdown_gen_profit,
			 PO *pp_arg_max,
			 PO *is_shutdown_profitable,
			 PO *is_startup_profitable,
			 int nSize) {
  cp(startup_sp_in); cp(fixed_and_fuel_startup_cost);
  cp(shutdown_gen_profit); cp(pp_arg_max); cp(is_shutdown_profitable);
  cp(is_startup_profitable);

  is_start_profitable_internal((double *) cn(startup_sp_in),
			       shutdown_sp_in,
			       (double *) cn(fixed_and_fuel_startup_cost),
			       xud_startup_horizon,
			       xud_shutdown_horizon,
			       max_cap,
			       (double *) cn(shutdown_gen_profit),
			       (double *) cn(pp_arg_max),
			       (short *) cn(is_shutdown_profitable),
			       (short *) cn(is_startup_profitable),
			       nSize);
}


void startup_cost_internal(short *is_cold_start,
			   short *starts,
			   double startup_cost_cold,
			   double startup_cost_p,
			   double *fuel_prices,
			   double start_fuel_cold,
			   double start_fuel,
			   double *res,
			   int nSize) {

  size_t idx;
  for (idx=0; idx < nSize; idx += 1)
    res[idx] = starts[idx] ? (is_cold_start[idx] ?
			      startup_cost_cold + fuel_prices[idx] * start_fuel_cold :
			      startup_cost_p + fuel_prices[idx] * start_fuel) :
      0.;
}


void startup_cost(PO *is_cold_start,
		  PO *starts,
		  double startup_cost_cold,
		  double startup_cost_p,
		  PO *fuel_prices,
		  double start_fuel_cold,
		  double start_fuel,
		  PO *res,
		  int nSize) {
  cp(is_cold_start); cp(starts); cp(fuel_prices); cp(res);
  startup_cost_internal( (short*) cn(is_cold_start),
			 (short *) cn(starts),
			 startup_cost_cold,
			 startup_cost_p,
			 (double *) cn(fuel_prices),
			 start_fuel_cold,
			 start_fuel,
			 (double *) cn(res),
			 nSize);
}
		  
