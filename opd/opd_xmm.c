#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL opd_xmm

#include <stdint.h>
#include <emmintrin.h>
#include <python2.7/Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include "opd_xmm.h"

#define reg __m128d 
#define madd _mm_add_pd
#define mmul _mm_mul_pd
#define mset _mm_set_pd1
#define mloa _mm_load_pd
#define msto _mm_store_pd
#define msub _mm_sub_pd

#define cp(name) PyArrayObject *npy_## name = (PyArrayObject *) (name)

void add4_internal(double *r, double *a, double *b, double *c, double *d, double *y, 
		   int nSize) {
  // computes r - (a+b+c+d)
  size_t idx;
  __m128d a_xmm;
  __m128d b_xmm;
  __m128d c_xmm;
  __m128d d_xmm;
  __m128d r_xmm;
  for (idx=0; idx<nSize; idx += 2) {
    a_xmm = _mm_load_pd(a + idx);
    b_xmm = _mm_load_pd(b + idx);
    c_xmm = _mm_load_pd(c + idx);
    d_xmm = _mm_load_pd(d + idx);
    r_xmm = _mm_load_pd(r + idx);
    a_xmm = _mm_add_pd(a_xmm, b_xmm);
    c_xmm = _mm_add_pd(c_xmm, d_xmm);
    a_xmm = _mm_add_pd(a_xmm, c_xmm);
    a_xmm = _mm_sub_pd(r_xmm, a_xmm);
    _mm_store_pd(y+idx, a_xmm);
  }
}

void add4(PyObject *r, PyObject *a, PyObject *b, PyObject *c, PyObject *d, PyObject *y,
	  int n) {
  cp(r);
  cp(a);
  cp(b);
  cp(c);
  cp(d);
  cp(y);
  add4_internal((double *) npy_r->data,
		(double *) npy_a->data,
		(double *) npy_b->data,
		(double *) npy_c->data,
		(double *) npy_d->data,
		(double *) npy_y->data,
		n);
}

void mul4_internal(double *r, double *a, double *b, double c, double *y, 
		   int nSize) {
  // computes: r * a * b * c (c scalar, all other vectors)
  size_t idx;
  __m128d r_xmm;
  __m128d a_xmm;
  __m128d b_xmm;
  __m128d c_xmm = _mm_set_pd1(c);
  for (idx=0; idx<nSize; idx += 2) {
    a_xmm = _mm_load_pd(a + idx);
    b_xmm = _mm_load_pd(b + idx);
    r_xmm = _mm_load_pd(r + idx);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, b_xmm);
    a_xmm = _mm_add_pd(a_xmm, c_xmm);
    _mm_store_pd(y+idx, a_xmm);
  }
}

void mul4(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y,
	  int n) {
  cp(r);
  cp(a);
  cp(b);
  cp(y);
  mul4_internal((double *) npy_r->data,
		(double *) npy_a->data,
		(double *) npy_b->data,
		c,
		(double *) npy_y->data,
		n);
}

void mul5_internal(double *r, double *a, double *b, double c, double *y, 
		   int nSize) {
  // computes: r * a * b * c (c scalar, all other vectors)
  size_t idx;
  __m128d r_xmm;
  __m128d a_xmm;
  __m128d b_xmm;
  __m128d c_xmm = _mm_set_pd1(c);
  for (idx=0; idx<nSize; idx += 2) {
    a_xmm = _mm_load_pd(a + idx);
    b_xmm = _mm_load_pd(b + idx);
    r_xmm = _mm_load_pd(r + idx);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, b_xmm);
    a_xmm = _mm_add_pd(a_xmm, c_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    a_xmm = _mm_add_pd(a_xmm, r_xmm);
    _mm_store_pd(y+idx, a_xmm);
  }
}

void mul5(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y,
	  int n) {
  cp(r);
  cp(a);
  cp(b);
  cp(y);
  mul5_internal((double *) npy_r->data,
		(double *) npy_a->data,
		(double *) npy_b->data,
		c,
		(double *) npy_y->data,
		n);
}

void mul6_internal(double *r, double *a, double *b, double c, double *y, 
		   int nSize) {
  // computes: r * a * b * c (c scalar, all other vectors)
  size_t idx;
  __m128d a_xmm;
  __m128d c_xmm = _mm_set_pd1(c);
  for (idx=0; idx<nSize; idx += 2) {
    a_xmm = _mm_add_pd(a_xmm, c_xmm);
    a_xmm = _mm_add_pd(a_xmm, c_xmm);
  }
}

void mul6(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y,
	  int n) {
  cp(r);
  cp(a);
  cp(b);
  cp(y);
  mul6_internal((double *) npy_r->data,
		(double *) npy_a->data,
		(double *) npy_b->data,
		c,
		(double *) npy_y->data,
		n);
}

void mul7_internal(double *r, double *a, double *b, double c, double *y, 
		   int nSize) {
  size_t idx;
  __m128d a_xmm;
  __m128d c_xmm = _mm_set_pd1(c);
  __m128d b_xmm = _mm_set_pd1(c+1);
  __m128d d_xmm = _mm_set_pd1(c+2);
  __m128d e_xmm = _mm_set_pd1(c+3);
  for (idx=0; idx<nSize; idx += 2) {
    a_xmm = _mm_load_pd(a + idx);
    __m128d res_xmm = madd(mmul(madd(a_xmm, b_xmm),c_xmm),d_xmm);
    _mm_store_pd(y+idx, res_xmm);
  }
}

void mul7(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y,
	  int n) {
  cp(r);
  cp(a);
  cp(b);
  cp(y);
  mul7_internal((double *) npy_r->data,
		(double *) npy_a->data,
		(double *) npy_b->data,
		c,
		(double *) npy_y->data,
		n);
}




void skew_fom_internal(double F, double *delta_X, 
		       double c1_ch, double qv, 
		       double c2_ch, double c3_ch, 
		       double *sim_fom, 
		       size_t nb_sim) {
  size_t idx;
  reg F_xmm = mset(F);
  reg one_xmm = mset(1.);
  reg qv_xmm = mset(qv);
  reg c1_ch_xmm = mset(c1_ch);
  reg c2_ch_xmm = mset(c2_ch);
  reg c3_ch_xmm = mset(c3_ch);
  reg fact_xmm = mset(-3.);
  for (idx=0; idx< nb_sim; idx += 2) {
    reg delta_X_xmm = mloa(delta_X + idx);
    reg delta_X2_xmm = mmul(delta_X_xmm, delta_X_xmm);
    reg delta_X3_xmm = mmul(delta_X2_xmm, delta_X_xmm);
    reg delta_X4_xmm = mmul(delta_X3_xmm, delta_X_xmm);
    reg ft_xmm = msub(delta_X2_xmm, qv_xmm);
    // first term 
    ft_xmm = mmul(c1_ch_xmm, ft_xmm);
    // second term
    reg st_xmm = mmul(delta_X_xmm, qv_xmm);
    st_xmm = mmul(fact_xmm, st_xmm);
    st_xmm = madd(delta_X3_xmm, st_xmm);
    st_xmm = mmul(c2_ch_xmm, st_xmm);
    //third term 
    reg tt_xmm = mset(3.);
    tt_xmm = mmul(tt_xmm, qv_xmm);
    tt_xmm = mmul(tt_xmm, qv_xmm);
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

void skew_fom(double F, PyObject *delta_X, 
	      double c1_ch, double qv, double c2_ch, double c3_ch, 
	      PyObject *sim_fom, 
	      size_t nb_sim) {
  cp(delta_X);
  cp(sim_fom);
  skew_fom_internal(F, (double *) npy_delta_X->data, 
		    c1_ch, qv, c2_ch, c3_ch, 
		    (double *) npy_sim_fom->data, 
		    nb_sim);
  
}

void skew_fom_test(double F, double *delta_X, 
		   double c1_ch, double qv, 
		   double c2_ch, double c3_ch, 
		   double *sim_fom, 
		   size_t nb_sim) {
  size_t idx;
  for (idx=0; idx< nb_sim; idx += 1) {
    double delta_X_xmm = delta_X[idx];
    double delta_X2_xmm = delta_X_xmm * delta_X_xmm;
    double delta_X3_xmm = delta_X2_xmm * delta_X_xmm;
    double delta_X4_xmm = delta_X3_xmm * delta_X_xmm;
    double ft_xmm = delta_X2_xmm - qv;
    // first term 
    ft_xmm = c1_ch * ft_xmm;
    // second term
    double st_xmm = c2_ch * (delta_X3_xmm -3. * delta_X_xmm * qv);
    //third term 
    double tt_xmm = 3. * qv*qv;
    // delta_X3 overwritten
    delta_X3_xmm = c3_ch * (delta_X3_xmm + delta_X4_xmm -6. * qv * delta_X2_xmm);
    // result
    double res_xmm = F * (1. + delta_X_xmm + ft_xmm + st_xmm + tt_xmm);
    sim_fom[idx] = res_xmm;
  }
}


// performs np.sum(vec1 * vec2)
double num_quad_internal(double *vec1, double *vec2, size_t v_len) {
  reg v1_xmm, v2_xmm, v3_xmm, res_xmm;
  size_t idx;
  double res[2];
  res_xmm = _mm_setzero_pd();
  for (idx=0; idx < v_len; idx += 2) {
    v1_xmm = mloa(vec1 + idx);
    v2_xmm = mloa(vec2 + idx);
    v2_xmm = mmul(v1_xmm, v2_xmm);  // multiplied 
    res_xmm = madd(res_xmm, v2_xmm); // added 
  }
  msto(res, res_xmm);
  return res[0] + res[1];
}

double num_quad(PyObject *vec1, PyObject *vec2, int v_len) {
  cp(vec1);
  cp(vec2);
  return num_quad_internal((double *) npy_vec1->data,
			   (double *) npy_vec2->data,
			   (size_t) v_len);
}
