#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL opd_xmm
#define AVX2

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#ifdef AVX2
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif

#define PO PyObject 

#include <python2.7/Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include "vtpm_cpu.h"

#ifdef XMM
#define DOUBLE_INCR 2
#define reg __m128d
#define madd _mm_add_pd
#define mmul _mm_mul_pd
#define mset _mm_set_pd1
#define mloa _mm_load_pd
#define msto _mm_store_pd
#define msub _mm_sub_pd
#define mmax _mm_max_pd
#endif

#ifdef AVX2
#define DOUBLE_INCR 4
#define reg __m256d
#define reg_int __m256i
#define madd _mm256_add_pd
#define mmul _mm256_mul_pd
#define mset _mm256_set1_pd
#define msetz _mm256_setzero_pd
#define mloa _mm256_loadu_pd
#define mloa_int _mm256
#define msto _mm256_storeu_pd
#define msub _mm256_sub_pd
#define mmax _mm256_max_pd
#endif

#define cp(name) PyArrayObject *npy_## name = (PyArrayObject *) (name)

void vm_add(double *x, double y, int nSize) {
  // computes x += y
  // nSize has to be divisible by 4
  size_t idx;
  reg x_xmm, y_xmm;
  y_xmm = mset(y);  
  for (idx=0; idx<nSize; idx += DOUBLE_INCR) {
    x_xmm = mloa(x + idx);
    x_xmm = madd(x_xmm, y_xmm);
    msto(x+idx, x_xmm);
  }
}


void vm_mul(double *x, double y, int nSize) {
  // computes x *= y
  // nSize has to be divisible by 4
  size_t idx;
  reg x_xmm, y_xmm;
  y_xmm = mset(y);  
  for (idx=0; idx<nSize; idx += DOUBLE_INCR) {
    x_xmm = mloa(x + idx);
    x_xmm = mmul(x_xmm, y_xmm);
    msto(x+idx, x_xmm);
  }
}

void vm_mul_omp(double *v, double *m, int n_rows, int n_cols) {
  int idx;
#pragma omp parallel for private(idx) shared(v,m)
  for (idx=0; idx<n_rows; idx +=1)
    vm_mul(m + n_cols * idx, v[idx], n_cols);
}

void vm_mul_py(PO *v, PO *m, int n_rows, int n_cols) {
  // compute m *= v, where v is a column vector 
  cp(v); cp(m);
  vm_mul_omp((double *) npy_v->data, (double *) npy_m->data, n_rows, n_cols);
}


// vm_ao doing the work 
void vm_ao_do(double *prev, double a1, double m1, double *sim, double *next,
	      int n_cols) {
  // computes next = prev + a1 + m1 * sim, for the entire row 
  // n_cols has to be divisible by 4
  size_t idx;
  reg prev_xmm, a1_xmm, m1_xmm, sim_xmm, next_xmm;
  a1_xmm = mset(a1);
  m1_xmm = mset(m1);
  for (idx=0; idx<n_cols; idx += DOUBLE_INCR) {
    prev_xmm = mloa(prev + idx);
    sim_xmm = mloa(sim+idx);
    next_xmm = mmul(m1_xmm, sim_xmm);
    next_xmm = madd(a1_xmm, next_xmm);
    next_xmm = madd(prev_xmm, next_xmm);
    msto(next+idx,next_xmm);
  }
}


void vm_ao_omp(double *prev, double *a1, double *m1, double *sim, double *next,
	       int n_rows, int n_cols) {
  int idx;
#pragma omp parallel for private(idx) shared(prev, a1, m1, sim, next)
  for (idx=0; idx<n_rows; idx +=1) {
    int start_row = n_cols * idx;
    vm_ao_do(prev + start_row, a1[idx], m1[idx], sim + start_row,
	     next + start_row, n_cols);
  }
}


void vm_ao(PO *prev, PO *a1, PO *m1, PO *sim, PO *next, int n_rows, int n_cols) {
  // computes res = prev + a1 + m1 * sim
  // on columns 
  cp(prev); cp(a1); cp(m1); cp(next); cp(sim);
  vm_ao_omp((double *) npy_prev->data, (double *) npy_a1->data,
	    (double *) npy_m1->data, (double *) npy_sim->data,
	    (double *) npy_next->data, n_rows, n_cols);
}


// maximum of two matrices
void max2m_do(double *m1, double *m2, double *res, int n_cols) {
  // computes next = prev + a1 + m1 * sim, for the entire row 
  // n_cols has to be divisible by 4
  size_t idx;
  reg m1_xmm, m2_xmm, res_xmm;
  for (idx=0; idx<n_cols; idx += DOUBLE_INCR) {
    m1_xmm = mloa(m1 + idx);
    m2_xmm = mloa(m2 + idx);
    res_xmm = mmax(m1_xmm, m2_xmm);
    msto(res+idx, res_xmm);
  }
}


void max2m_omp(double *m1, double *m2, double *res, int n_rows, int n_cols) {
  int idx;
#pragma omp parallel for private(idx) shared(m1, m2, res)
  for (idx=0; idx<n_rows; idx +=1) {
    int start_row = n_cols * idx;
    max2m_do(m1 + start_row, m2 + start_row, res + start_row, n_cols);
  }
}


void max2m(PO *m1, PO *m2, PO *res, int n_rows, int n_cols) {
  // computes np.maximum(m1, m2)
  cp(m1); cp(m2); cp(res);
  max2m_omp((double *) npy_m1->data, (double *) npy_m2->data,
	    (double *) npy_res->data, n_rows, n_cols);
}
