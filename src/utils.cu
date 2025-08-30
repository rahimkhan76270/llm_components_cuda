#include "utils.cuh"
#include <curand.h>

void GenerateRandomNumCuRand(float* data,int64_t nums,int64_t seed) {
    float *data_d;
    cudaMalloc(&data_d,nums*sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,seed);
    curandGenerateUniform(gen,data_d,nums);
    cudaMemcpy(data,data_d,nums*sizeof(float),cudaMemcpyDefault);
    cudaFree(data_d);
    curandDestroyGenerator(gen);
}

void GenerateRandomIntegerCuRand(int64_t* data,int64_t nums,int64_t seed) {
    unsigned int* data_d;
    unsigned int *data1 = new unsigned int[nums];
    cudaMalloc(&data_d,nums*sizeof(unsigned int));
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,seed);
    curandGeneratePoisson(gen,data_d,nums,seed);
    cudaMemcpy(data1,data_d,nums*sizeof(unsigned int),cudaMemcpyDefault);
    cudaFree(data_d);
    curandDestroyGenerator(gen);
    for (int64_t i=0;i<nums;i++) data[i]=static_cast<int64_t>(data1[i]);
    free(data1);
}