#include "embedding.cuh"
__global__ void GetEmbeddingKernel(const float * EmbeddingWeights,const int64_t* tokens,float * output,int64_t num_tokens,int64_t batch_size,int64_t embedding_dim) {
    // embedding weights ia a 2D matrix of the size vocab_size X embedding_dim
    // tokens is the list of list of integers representing batch of tokens
    // output matrix of embeddings of size batch_size x num_tokens x embedding_dim
    int64_t row =blockIdx.y*blockDim.y+threadIdx.y;
    int64_t col=blockIdx.x*blockDim.x+threadIdx.x;
    int64_t idx=row*num_tokens+col;
    if (row<batch_size && col<num_tokens) {
        int64_t token=tokens[idx];
        for (int i=0;i<embedding_dim;i++) {
            output[i*1+col*embedding_dim+row*embedding_dim*num_tokens]=EmbeddingWeights[token*embedding_dim+i];
        }
    }
}

void GetEmbedding(const float * EmbeddingWeights,const int64_t* tokens,float * output,int64_t num_tokens,int64_t batch_size,int64_t embedding_dim) {
    dim3 threads(16,16);
    dim3 blocks((batch_size+threads.x-1)/threads.x,(num_tokens+threads.y-1)/threads.y);
    GetEmbeddingKernel<<<blocks,threads>>>(EmbeddingWeights,tokens,output,num_tokens,batch_size,embedding_dim);
    cudaDeviceSynchronize();
}
