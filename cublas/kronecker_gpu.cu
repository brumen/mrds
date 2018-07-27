#include <thrust/device_vector.h>
#include "kronecker_gpu.h"

void *__dso_handle = NULL;

struct add_functor {
  const float a;
  add_functor(float _a) : a(_a) {}  // constructor
  __host__ __device__ 
  float operator()(const float& x, const float &y) const {
    return a + x;
  }
};

struct multiply_functor {
  const float a;
  multiply_functor(float _a) : a(_a) {}  // constructor
  __host__ __device__ 
  float operator()(const float& x, const float &y) const {
    return a * x;
  }
};

void vpv_simple(thrust::device_vector<float> &v1, 
 		thrust::device_vector<float> &v2,
 		thrust::device_vector<float> &m_new) {
  // constructs v1 + v2 = m_new 
  int row_idx;
  int v2_size = v2.size();
  float v1_used;
  thrust::device_vector<float>::iterator where_start;
  for (row_idx=0; row_idx < v1.size(); row_idx += 1) {
    v1_used = v1[row_idx];
    // m_new[row_idx, :] = v1[row_idx] + v2[row_idx, :];
    where_start = (thrust::device_vector<float>::iterator) (m_new.begin() + row_idx * v2_size);
    thrust::transform(v2.begin(), v2.end(),
    		      where_start, where_start,
    		      add_functor(v1_used));
  }
}

void vpv(float *v1_p, float *v2_p, float *m_new_p, 
	 int v1_size, int v2_size) {
  thrust::device_ptr<float> v1_dp = thrust::device_pointer_cast(v1_p);
  thrust::device_ptr<float> v2_dp = thrust::device_pointer_cast(v2_p);
  thrust::device_ptr<float> m_new_dp = thrust::device_pointer_cast(m_new_p);
  thrust::device_vector<float> v1(v1_dp, v1_dp + v1_size); 
  thrust::device_vector<float> v2(v2_dp, v2_dp + v2_size);
  thrust::device_vector<float> m_new(m_new_dp, m_new_dp + v1_size * v2_size);
  vpv_simple(v1, v2, m_new);
}


void vtv_simple(thrust::device_vector<float> &v1, 
 		thrust::device_vector<float> &v2,
 		thrust::device_vector<float> &m_new) {
  // constructs v1 * v2 = m_new 
  int row_idx;
  int v2_size = v2.size();
  float v1_used;
  thrust::device_vector<float>::iterator where_start;
  for (row_idx=0; row_idx < v1.size(); row_idx += 1) {
    v1_used = v1[row_idx];
    // m_new[row_idx, :] = v1[row_idx] * v2[row_idx, :];
    where_start = (thrust::device_vector<float>::iterator) (m_new.begin() + row_idx * v2_size);
    thrust::transform(v2.begin(), v2.end(),
    		      where_start, where_start,
    		      multiply_functor(v1_used));
  }
}
