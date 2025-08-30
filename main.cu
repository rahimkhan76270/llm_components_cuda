#include <iostream>
#include "utils.cuh"
#include "embedding.cuh"

int main() {
    int64_t vocab_size=1000;
    int64_t embedding_dim=512;
    int64_t num_tokens=100;
    int64_t batch_size=2;
    auto *tokens=new int64_t[batch_size*num_tokens];
    auto *embedding_matrix=new float[vocab_size*embedding_dim];
    auto *output_emeds=new float[batch_size*num_tokens*embedding_dim];
    float *embedding_matrix_d,*output_embeds_d;
    int64_t *tokens_d;
    GenerateRandomIntegerCuRand(tokens,batch_size*num_tokens,12);
    GenerateRandomNumCuRand(embedding_matrix,vocab_size*embedding_dim,12);
    cudaMalloc(&embedding_matrix_d,vocab_size*embedding_dim*sizeof(float));
    cudaMalloc(&tokens_d,batch_size*num_tokens*sizeof(int64_t));
    cudaMalloc(&output_embeds_d,batch_size*num_tokens*embedding_dim*sizeof(float));
    cudaMemcpy(embedding_matrix_d,embedding_matrix,vocab_size*embedding_dim*sizeof(float),cudaMemcpyDefault);
    GetEmbedding(embedding_matrix_d,tokens_d,output_embeds_d,num_tokens,batch_size,embedding_dim);
    cudaMemcpy(output_emeds,output_embeds_d,batch_size*num_tokens*embedding_dim*sizeof(float),cudaMemcpyDefault);
    cudaFree(embedding_matrix_d);
    cudaFree(tokens_d);
    cudaFree(output_embeds_d);
    for (int i=0;i<batch_size*num_tokens;i++) {
        std::cout<<tokens[i]<<" ";
    }
    std::cout<<std::endl;
    // for (int row=0;row<vocab_size;row++) {
    //     for (int col=0;col<embedding_dim;col++) {
    //         std::cout<<embedding_matrix[row*embedding_dim+col]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    std::cout<<std::endl;
    free(tokens);
    free(embedding_matrix);
    free(output_emeds);
    return 0;
}
