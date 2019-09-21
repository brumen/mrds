#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL opd_xmm
#define AVX2  // switches between XMM and AVX

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef AVX2
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif

#define PO PyObject

#include <python3.5m/Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include "opd_avx.h"


#ifdef XMM
#define DOUBLE_INCR 2
#define SHORT_INCR 8
#define reg     __m128d
#define reg_int __m128i
#define madd    _mm_add_pd
#define mmul    _mm_mul_pd
#define mset    _mm_set_pd1
#define mseti   _mm_set1_epi16  // setting of integers
#define msetz   _mm_setzero_pd
#define mloa    _mm_load_pd
#define msto    _mm_store_pd
#define msub    _mm_sub_pd
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

#define CP(name) PyArrayObject *npy_## name = (PyArrayObject *) (name)  // mnemonic for change pointer
#define CN(name) npy_## name  // mnemonic for change name
#define CPN(typeused, name) typeused *npy_##name = (typeused *) ((PyArrayObject *)(name))->data

void add4(PO *r, PO *a, PO *b, PO *c, PO *d, PO *y, int n) {
  // computes r - (a+b+c+d)
  // Using g++-8 the compiler opimizes this using vaddpd instructions.

  CPN(double, r);
  CPN(double, a);
  CPN(double, b);
  CPN(double, c);
  CPN(double, d);
  CPN(double, y);

  for (size_t idx=0; idx<n; idx += 1)
    CN(y)[idx] = CN(r)[idx] - (CN(a)[idx] + CN(b)[idx] + CN(c)[idx] + CN(d)[idx]);
}


void mul4(PO *r, PO *a, PO *b, double c, PO *y, int n) {
  // computes: r * a * b * c (c scalar, all other vectors)
  CPN(double, r);
  CPN(double, a);
  CPN(double, b);
  CPN(double, y);

  for (size_t idx=0; idx<n; idx += 1)
    CN(y)[idx] = CN(r)[idx] * CN(a)[idx] * CN(b)[idx] * c;

}

void skew_fom(double F
              , PO *delta_X
              , double c1_ch
              , double qv
              , double c2_ch
              , double c3_ch
              , PO *sim_fom
              , int nb_sim) {

  CPN(double, delta_X);
  CPN(double, sim_fom);

  for (size_t idx = 0; idx< nb_sim; idx += 1) {
    double delta_X_xmm  = CN(delta_X)[idx];
    double delta_X2_xmm = delta_X_xmm * delta_X_xmm;

    CN(sim_fom)[idx] =  F * (1.
                             + delta_X_xmm
                             + c1_ch * delta_X2_xmm - qv
                             + c2_ch * (delta_X2_xmm * delta_X_xmm  -3. * delta_X_xmm * qv)
                             + c3_ch * (delta_X2_xmm * delta_X2_xmm -6. * qv * delta_X2_xmm + 3. * qv*qv));
  }

}



double num_quad_internal(double *vec1, double *vec2, size_t v_len) {
  // performs np.sum(vec1 * vec2), a scalar product
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
  CP(vec1); CP(vec2);
  return num_quad_internal((double *) CN(vec1),
                           (double *) CN(vec2),
                           (size_t) v_len);
}


double num_quad_internal_simple(double *vec1, double *vec2, size_t v_len) {
  // performs np.sum(vec1 * vec2), a scalar product

  double res = 0.;
  for (size_t idx=0; idx < v_len; idx += 1)
    res += vec1[idx] * vec2[idx];

  return res;
}

double num_quad_simple(PO *vec1, PO *vec2, int v_len) {
  CP(vec1); CP(vec2);

  return num_quad_internal_simple((double *) CN(vec1),
                                  (double *) CN(vec2),
                                  (size_t) v_len);
}


void do_start_shut_internal(short *dc_can,
			                short *dc_force,
			                short *is_profitable,
			                short *do_action,
			                int   nb) {
  // implements startup/shutdown
  // dc_can_start & ((dc_force_start == 2) || (is_profit & dc_force == 1))
  // and stores it into do_action

  size_t idx;
  reg_int *dc_can_avx        = (reg_int *) dc_can;
  reg_int *dc_force_avx      = (reg_int *) dc_force;
  reg_int *is_profitable_avx = (reg_int *) is_profitable;
  reg_int *do_action_avx     = (reg_int *) do_action;

  for (idx=0; idx < nb/SHORT_INCR; idx += 1) {
    reg_int can_avx     = _mm256_loadu_si256(dc_can_avx + idx);
    reg_int force_avx   = _mm256_loadu_si256(dc_force_avx + idx);
    reg_int is_prof_avx = _mm256_loadu_si256(is_profitable_avx + idx);

    reg_int cnd_1 = _mm256_cmpeq_epi16(force_avx, _mm256_set1_epi16(2));
    reg_int cnd_2 = _mm256_and_si256(is_prof_avx,
				     _mm256_cmpeq_epi16(force_avx, _mm256_set1_epi16(1)));

    cnd_1 = _mm256_or_si256(cnd_1, cnd_2);
    cnd_1 = _mm256_and_si256(can_avx, cnd_1);

    _mm256_storeu_si256(do_action_avx + idx, cnd_1);
  }
}


void do_start_shut(PO *dc_can, PO *dc_force, PO *is_profitable, PO *do_action, int nb_sim) {
  // Implements whether one can start/shut down the plant.
  CP(dc_can);
  CP(dc_force);
  CP(is_profitable);
  CP(do_action);

  do_start_shut_internal((short *) CN(dc_can),
                         (short *) CN(dc_force),
                         (short *) CN(is_profitable),
                         (short *) CN(do_action),
                         nb_sim);
}

void do_start_shut_simple(PO *dc_can
                          , PO *dc_force
                          , PO *is_profitable
                          , PO *do_action
                          , int nb_sim) {
  CPN(bool , dc_can);
  CPN(short, dc_force);
  CPN(bool , is_profitable);
  CPN(bool, do_action);

  for (size_t idx=0; idx < nb_sim; idx += 1)
    CN(do_action)[idx] = CN(dc_can)[idx] & ( (CN(dc_force)[idx] == 2) || (CN(is_profitable)[idx] & (CN(dc_force)[idx] == 1)));

}


void cold_start(PO *hours_shut,
                PO *res,
                short xud_cold_start,
                size_t nSize) {
  //
  // WHAT IS IMPLEMNTING HERE???

  CPN(short, hours_shut);  // creates npy_hours_shut
  CPN(short, res);

  reg_int cold_start_avx = mseti(xud_cold_start);
  reg_int *hours_shut_avx = (reg_int *) hours_shut;
  reg_int *res_avx  = (reg_int *) res;
  for (size_t idx=0; idx < nSize / SHORT_INCR; idx +=1) {
    reg_int hs_avx = _mm256_load_si256(hours_shut_avx + idx);
    reg_int CNd_1 = _mm256_cmpgt_epi16(hs_avx, cold_start_avx);
    _mm256_store_si256(res_avx+idx, CNd_1);
  }
  // Missing if nSize not divisible by 16 ????

  // for (idx = nSize/SHORT_INCR; idx < nSize; idx +=1)


}

// selecting from 2 doubles according to int16
void sel_double2_internal(double *x1,
                          double *x2,
                          short  *sel,
                          double *res,
                          int n ) {
  // implements:
  // if sel == 0xFFFF x1, else x2
  reg res_avx;

  for (size_t idx=0; idx<n; idx+= 2) {
    reg sel_avx   = _mm256_shuffle_pd( (reg) mseti(sel[idx])
                                     , (reg) mseti(sel[idx+1])
                                     , _MM_SHUFFLE2(0,0));
    reg x1_avx    = mloa(x1+idx);

    msto(res+idx, _mm256_xor_pd(x1_avx
                                , _mm256_and_pd(sel_avx
                                                ,_mm256_xor_pd(mloa(x2+idx), x1_avx););
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

  reg xud_start_avx  = mset(xud_startup_horizon);
  reg xud_shut_avx   = mset(xud_shutdown_horizon);
  reg max_cap_avx    = mset(max_cap);
  reg sh_max_cap_avx = mset(1./(xud_startup_horizon * max_cap));
  reg shut_sp_in_avx = mset(shutdown_sp_in);

  for (size_t idx=0; idx < nSize; idx += 2) {
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
  CP(startup_sp_in);
  CP(fixed_and_fuel_startup_cost);
  CP(shutdown_gen_profit);
  CP(pp_arg_max);
  CP(is_shutdown_profitable);
  CP(is_startup_profitable);

  is_start_profitable_internal((double *) CN(startup_sp_in),
			       shutdown_sp_in,
			       (double *) CN(fixed_and_fuel_startup_cost),
			       xud_startup_horizon,
			       xud_shutdown_horizon,
			       max_cap,
			       (double *) CN(shutdown_gen_profit),
			       (double *) CN(pp_arg_max),
			       (short *) CN(is_shutdown_profitable),
			       (short *) CN(is_startup_profitable),
			       nSize);
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

  CPN(short,  is_cold_start);  // constructs npy_is_cold_start of type short
  CPN(short,  starts);
  CPN(double, fuel_prices);
  CPN(double, res);

  for (size_t idx=0; idx < nSize; idx += 1)
    npy_res[idx] = npy_starts[idx] ? (npy_is_cold_start[idx] ?
                                      startup_cost_cold + npy_fuel_prices[idx] * start_fuel_cold :
                                      startup_cost_p + npy_fuel_prices[idx] * start_fuel) :
      0.;

}
