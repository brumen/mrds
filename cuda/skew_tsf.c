// transformation code 
__global__ void F_skew_tsf( float F
                          , float c1
                          , float c2
                          , float c3
                          , float V_u
                          , float *X_u
                          , float *res
                          , int n ) {
  // implements:
  // F_res = F_u * (1. + X_u + 0.5 * c1 * (X_u**2 - V_u) +
  //                                     c2 * (X_u**3 - 3. * X_u * V_u) / 6. +
  //                                     c3 * (X_u**4 - 6. * V_u * X_u**2 + 3. * V_u**2) / 24.)

  int tx = blockIdx.x;
  float xu = X_u[tx];
  float vu = V_u;
  float xu2, xu3, xu4, vuxu, vuxu2;
  float r1 = 1.;
  
  if (tx < n) {
    xu2 = xu * xu; 
    xu3 = xu2 * xu;
    xu4 = xu3 * xu;
    vuxu = xu * vu;
    vuxu2 = vuxu * xu;

    res[tx] = F * ( 1. + xu + 0.5 * c1 * (xu2 - vu) + c2 * (xu3 - 3. * vuxu) / 6. + c3 * (xu4 - 6. * vuxu2 + 3. * vu*vu) / 24.);
  }
}
