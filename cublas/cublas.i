%module cublas_i

%{
#include <python2.7/Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

  // has to be programmed in this fashion 
  int cublasIsamax2_d(long long int Ap,  int s) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasStatus_t stat3;
    cublasHandle_t handle;
    int result; // = 7;
    stat = cublasCreate_v2(&handle); 
    stat3 = cublasIsamax_v2(handle, s, (const float *) Ap, 1, &result);
    return result;
  }

  void cublasSgemv_d(long int Ap,  long int vp,
		     long int res,
		     int nb_rows, int nb_cols) {
    // m, n ... rows, column of Ap 
    // cudaError_t cudaStat;
    cublasStatus_t stat, stat2;
    cublasHandle_t handle;
    cublasOperation_t trans = CUBLAS_OP_T;  // BECAUSE C AND PYTHON ARE DIFFERENT
    float alpha = 1.;
    float beta = 0.;
    stat = cublasCreate_v2(&handle); 
    stat2 = cublasSgemv_v2(handle,
			   trans,
			   nb_rows, nb_cols,
			   &alpha,
			   (float *) Ap,
			   nb_rows,
			   (float *) vp,
			   1,
			   &beta,
			   (float *) res,
			   1);
  }

  void cublasDgemv_d(long int Ap,  long int vp,
		     long int res,
		     int nb_rows, int nb_cols) {
    // m, n ... rows, column of Ap 
    // cudaError_t cudaStat;
    cublasStatus_t stat, stat2;
    cublasHandle_t handle;
    cublasOperation_t trans = CUBLAS_OP_T;  // BECAUSE C AND PYTHON ARE DIFFERENT
    double alpha = 1.;
    double beta = 0.;
    stat = cublasCreate_v2(&handle); 
    stat2 = cublasDgemv_v2(handle,
			   trans,
			   nb_rows, nb_cols,
			   &alpha,
			   (double *) Ap,
			   nb_rows,
			   (double *) vp,
			   1,
			   &beta,
			   (double *) res,
			   1);
  }

  
%}

int cublasIsamax2_d(long long int Ap,  int s);
void cublasSgemv_d(long int Ap, long int vp,
        		   long int res,
		           int nb_rows, int nb_cols);

void cublasDgemv_d(long int Ap, long int vp,
		   long int res,
		   int nb_rows, int nb_cols);
