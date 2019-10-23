/* 
   compile with: gcc -shared -fPIC tensor_prod.c -lpython2.7 -o tp.so 
   or look into Makefile 
*/

#include <math.h>
#include <stdio.h>
#include <python2.7/Python.h>
#include <numpy/arrayobject.h>

PyObject*  tensor_prod_2 (PyArrayObject *P_m, 
			  PyArrayObject *H_m, 
			  PyArrayObject *G_m, 
			  PyArrayObject *res_m ) {

  int F_1_ind, F_2_ind, F_1_out, F_2_out;
  int sh1 = P_m->dimensions[0];
  int sh2 = P_m->dimensions[1];
  int G_rows = G_m->dimensions[0];
  int G_cols = G_m->dimensions[1];

  double sum_tmp;

#pragma omp parallel for default(shared) private(sum_tmp, F_1_ind, F_2_ind, F_1_out, F_2_out)
  for (F_1_ind=0; F_1_ind < sh1; F_1_ind++) {
    for (F_2_ind=0; F_2_ind < sh2; F_2_ind++) {
      sum_tmp = 0.0;
      for (F_1_out=0; F_1_out < G_rows; F_1_out++) {
	for (F_2_out=0; F_2_out< G_cols; F_2_out++ ) {
	  sum_tmp += *(double *) (P_m->data + F_1_ind * P_m->strides[0] + 
				  F_2_ind * P_m->strides[1] + 
				  F_1_out * P_m->strides[2] + 
				  F_2_out * P_m->strides[3]) * 
	    *(double *) (H_m->data + F_1_out * H_m->strides[0] + 
			 H_m->strides[1] * F_2_out);
	}
      }
      *(double *)(res_m->data + F_1_ind * res_m->strides[0] + F_2_ind * res_m->strides[1] ) = sum_tmp + * (double *) ( G_m->data + F_1_ind * G_m->strides[0] + F_2_ind * G_m->strides[1] );
    }
  }

  return (PyObject *) res_m;
}

/* 
   check tensor_pp.py 
   (if it doesnt work, play around with this below )

   import ctypes
   ctypes.cdll.LoadLibrary("/home/brumen/workspace/mrds/tp.so")
*/
		    
