#include "positional_encodings.cuh"

__global__ void AbsolutePositionalEncodingsKernel(float* output,int64_t num_tokens,int64_t embedding_dim,int64_t d_model) {
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<num_tokens && col<embedding_dim) {
        int i = col / 2;
        float angle = (float)row / powf(10000.0f, 2.0f * i / d_model);
        output[row * embedding_dim + col] = (col % 2 == 0) ? __sinf(angle) : __cosf(angle);
    }
}

__global__ void  AddAbsolutePositionalEncodingsKernel(float *input,const float* pos_encs,int64_t num_tokens,int64_t embedding_dim,int64_t batch_size) {
    int batch=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.z*blockDim.z+threadIdx.z;
    if (batch<batch_size && row<num_tokens && col<embedding_dim) {
        input[col*1+row*embedding_dim+batch*embedding_dim*num_tokens]+=pos_encs[row*embedding_dim+col];
    }
}

void AddAbsolutePositionalEncodings(float *input,int64_t num_tokens,int64_t embedding_dim,int64_t d_model,int64_t batch_size) {
    float* pos_encodings;
    cudaMalloc(&pos_encodings,num_tokens*embedding_dim*sizeof(float));
    dim3 thread1(16,16);
    dim3 blocks1((num_tokens+thread1.x-1)/thread1.x,(embedding_dim+thread1.y-1)/thread1.y);
    AbsolutePositionalEncodingsKernel<<<blocks1,thread1>>>(pos_encodings,num_tokens,embedding_dim,d_model);
    cudaDeviceSynchronize();

    dim3 thread2(8,8,8);
    dim3 blocks2((batch_size+thread2.x-1)/thread2.x,(num_tokens+thread2.y-1)/thread2.y,(embedding_dim+thread2.z-1)/thread2.z);
    AddAbsolutePositionalEncodingsKernel<<<blocks2,thread2>>>(input,pos_encodings,num_tokens,embedding_dim,batch_size);
    cudaDeviceSynchronize();
    cudaFree(pos_encodings);
}