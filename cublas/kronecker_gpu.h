// m_new needs to be the size of v1.size * v2.size 
//void vpv_simple(thrust::device_vector<float> &v1, 
// 		thrust::device_vector<float> &v2,
// 		thrust::device_vector<float> &m_new);

void vpv(float *v1p, float *v2p, float *m_new_p, 
	 int v1_size, int v2_size);

//void vtv_simple(thrust::device_vector<float> &v1, 
// 		thrust::device_vector<float> &v2,
// 		thrust::device_vector<float> &m_new);
