%module curand

%{
#include "/usr/include/cuda.h"
#include "/usr/include/curand.h"

void gen_eff_dev_rns(int n, long long int devData_p, curandGenerator_t *gen) {
  curandGenerator_t *gen_used = (curandGenerator_t *) gen;
  float *devData = (float *) devData_p;
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); 
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A); 
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32); 
  curandSetPseudoRandomGeneratorSeed(*gen_used, 1234ULL);
  /* Generate n floats on device */ 
  curandGenerateNormal(*gen_used, devData, n, 0., 1.); 
}

void gen_eff_dev_rns_double(int n, long long int devData_p, curandGenerator_t *gen) {
  curandGenerator_t *gen_used = (curandGenerator_t *) gen;
  double *devData = (double *) devData_p;
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); 
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A); 
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32); 
  curandSetPseudoRandomGeneratorSeed(*gen_used, 1234ULL);
  /* Generate n doubles on device */ 
  curandGenerateNormalDouble(*gen_used, devData, n, 0., 1.); 
}


curandGenerator_t* create_gen_simple() {
  curandGenerator_t *gen = malloc(sizeof(curandGenerator_t));
  curandCreateGenerator(gen, CURAND_RNG_PSEUDO_MTGP32); 
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A); 
  return gen;
}

%}

%include "/usr/include/cuda.h"
%include "/usr/include/curand.h"
void gen_eff_dev_rns(int n, long long int devData_p, curandGenerator_t *gen);
void gen_eff_dev_rns_double(int n, long long int devData_p, curandGenerator_t *gen);
curandGenerator_t* create_gen_simple();
