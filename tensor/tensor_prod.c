/* gcc -shared -fPIC tensor_prod.c -lpython2.7 -lpthread -o tp.so */

#include <math.h>
#include <stdio.h>
/* #include <python2.7/Python.h> */
#include <python3.5/Python.h>
#include <python3.5/numpy/arrayobject.h>
/* #include<numpy/ndarraytypes.h> */
#include <pthread.h>

#define min(a,b)                                \
  ({ __typeof__ (a) _a = (a);                   \
    __typeof__ (b) _b = (b);                    \
    _a > _b ? _b : _a; })

/* structure to gather all the matrices together */
typedef struct {
  PyArrayObject *P_m;
  PyArrayObject *H_m;
  PyArrayObject *G_m;
  PyArrayObject *res_m;
  int shape1;
  int shape2;
  int thread_nb; /* number of the thread */
} tt_s; /* together tensors structure */


void* tensor_prod_partial (void *t_all) {

  int F_1_ind, F_2_ind, F_1_out, F_2_out;
  double sum_tmp;
  tt_s *ts = (tt_s *) t_all; /* casting of a pointer */
  int shape1_by_four = (int) ( (ts->shape1 + 3) / 4);

  /* H and G dimensions are the same */
  int G_rows = ts->G_m->dimensions[0];
  int G_cols = ts->G_m->dimensions[1];

  for (F_1_ind= (ts->thread_nb) * shape1_by_four;
       F_1_ind < min( (ts->thread_nb+1)*shape1_by_four , ts->shape1 );
       F_1_ind = F_1_ind + 1) {
    for (F_2_ind=0; F_2_ind < ts->shape2; F_2_ind++) {
      sum_tmp = 0.0;
      for (F_1_out=0; F_1_out < G_rows; F_1_out++) {
        for (F_2_out=0; F_2_out< G_cols; F_2_out++ ) {
          sum_tmp += *(double *) (ts->P_m->data + F_1_ind * ts->P_m->strides[0] +
                                  F_2_ind * ts->P_m->strides[1] +
                                  F_1_out * ts->P_m->strides[2] +
                                  F_2_out * ts->P_m->strides[3]) *
            *(double *) (ts->H_m->data + F_1_out * ts->H_m->strides[0] +
                         ts->H_m->strides[1] * F_2_out);
        }
      }
      *(double *)(ts->res_m->data + F_1_ind * ts->res_m->strides[0] +
                  F_2_ind * ts->res_m->strides[1] ) = sum_tmp + * (double *)
        ( ts->G_m->data + F_1_ind * ts->G_m->strides[0] + F_2_ind * ts->G_m->strides[1] );
    }
  }
}




/* tensor product with 4 threads 
   shape1 >= 4
*/
PyObject*  tensor_prod_2 (PyArrayObject *P_m,
                          PyArrayObject *H_m,
                          PyArrayObject *G_m,
                          PyArrayObject *res_m ) {

  pthread_t thread1, thread2, thread3, thread4;
  int iret1, iret2, iret3, iret4;

  int sh1 = P_m->dimensions[0];
  int sh2 = P_m->dimensions[1];

  tt_s t_all_0 = { P_m, H_m, G_m, res_m, sh1, sh2, 0 }; /* structure for all tensors */
  tt_s t_all_1 = { P_m, H_m, G_m, res_m, sh1, sh2, 1 };
  tt_s t_all_2 = { P_m, H_m, G_m, res_m, sh1, sh2, 2 };
  tt_s t_all_3 = { P_m, H_m, G_m, res_m, sh1, sh2, 3 };

  iret1 = pthread_create (&thread1, NULL, tensor_prod_partial, (void *) (&t_all_0));
  iret2 = pthread_create (&thread2, NULL, tensor_prod_partial, (void *) (&t_all_1));
  iret3 = pthread_create (&thread3, NULL, tensor_prod_partial, (void *) (&t_all_2));
  iret4 = pthread_create (&thread4, NULL, tensor_prod_partial, (void *) (&t_all_3));

  pthread_join (thread1, NULL);
  pthread_join (thread2, NULL);
  pthread_join (thread3, NULL);
  pthread_join (thread4, NULL);

  return (PyObject *) res_m;



}


/* version of tensor product for 1 thread */
PyObject*  tensor_prod_1 (PyArrayObject *P_m,
                          PyArrayObject *H_m,
                          PyArrayObject *G_m,
                          PyArrayObject *res_m ) {

  int F_1_ind, F_2_ind, F_1_out, F_2_out;
  int sh1 = P_m->dimensions[0];
  int sh2 = P_m->dimensions[1];
  double sum_tmp;

  for (F_1_ind=0; F_1_ind < sh1; F_1_ind++) {
    for (F_2_ind=0; F_2_ind < sh2; F_2_ind++) {
      sum_tmp = 0.0;
      for (F_1_out=0; F_1_out < sh1; F_1_out++) {
        for (F_2_out=0; F_2_out< sh2; F_2_out++ ) {
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
