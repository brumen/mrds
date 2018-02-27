#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL dsum_gorazd

#include <xmmintrin.h>
#include <python2.7/Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <mkl_vml_functions.h>
#include "dsum.h"

// MISSING EXTERN FUNCTIONS 

double dsum_double(double *pArray1, int nSize) {
  int nSize_mod = nSize % 4;
  double addon = 0.;
  double *pResult = (double *) malloc(2*sizeof(double));
  __m128d m0 = _mm_set_pd1(0.);
  __m128d pSrc1;
  size_t index;
  
  for (index=0; index < nSize; index += 2) {
    pSrc1 = _mm_load_pd(pArray1 + index);
    m0 = _mm_add_pd(pSrc1, m0);
  }
  _mm_store_pd(pResult, m0);
  switch (nSize_mod) {
  case 0:
    break;
  case 1:
    addon = *(pResult + nSize -1);
    break;
  }
  return *pResult + *(pResult+1) + addon;
}

/* expose function for summing to python */
double dsum_d(PyObject *x, int n) {
  PyArrayObject *npy_array_x = (PyArrayObject *) x;
  return dsum_double((double*) npy_array_x->data, n);
}

/* adds vector + scalar */
void vpscalar_double(double *a, double x, double *b, int nSize) {
  int nSize_mod = nSize % 2;
  double addon = 0.;
  double *pResult = (double *) malloc(2*sizeof(double));
  __m128d m0 = _mm_set_pd1(0.);
  __m128d m1 = _mm_set_pd1(x);
  __m128d pSrc1;
  size_t index;
  
  for (index=0; index < nSize; index += 2) {
    pSrc1 = _mm_load_pd(a + index);
    pSrc1 = _mm_add_pd(pSrc1, m1);
    _mm_store_pd(b + index, pSrc1);
  }
}

/* multiples vector * scalar */
void vmscalar_double(double *a, double x, double *b, int nSize) {
  double *pResult = (double *) malloc(2*sizeof(double));
  __m128d m1 = _mm_set_pd1(x);
  __m128d pSrc1;
  size_t index;
  
  for (index=0; index < nSize; index += 2) {
    pSrc1 = _mm_load_pd(a + index);
    pSrc1 = _mm_mul_pd(pSrc1, m1);
    _mm_store_pd(b + index, pSrc1);
  }
}

/* multiplication of two vectors */
void vdmul_py(PyObject *a, PyObject *b, PyObject *y, int n) {
  PyArrayObject *npy_a = (PyArrayObject *) a;
  PyArrayObject *npy_b = (PyArrayObject *) b;
  PyArrayObject *npy_y = (PyArrayObject *) y;
  vdmul_(&n, (double *) npy_a->data, (double *) npy_b->data, (double *) npy_y->data);
}

void vdadd_cons(PyObject *a, PyObject *b, PyObject *y) {
  PyArrayObject *npy_a = (PyArrayObject *) a;
  PyArrayObject *npy_b = (PyArrayObject *) b;
  PyArrayObject *npy_y = (PyArrayObject *) y;
  int n = npy_a->dimensions[0];

  vdadd_(&n, 

}
