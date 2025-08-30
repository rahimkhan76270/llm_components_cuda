#include <iostream>
#include <fmt/ranges.h>
#include "utils.cuh"
#include "embedding.cuh"
#include <vector>
int main() {
    int64_t vocab_size=50;
    int64_t embedding_dim=5;
    int64_t num_tokens=5;
    int64_t batch_size=2;
    std::vector<int64_t> tokens(batch_size*num_tokens);
    std::vector<float> embedding_matrix(vocab_size*embedding_dim);
    std::vector<float> output_embeds(batch_size*num_tokens*embedding_dim);
    float *embedding_matrix_d,*output_embeds_d;
    int64_t *tokens_d;
    GenerateRandomIntegerCuRand(tokens.data(),batch_size*num_tokens,12);
    GenerateRandomNumCuRand(embedding_matrix.data(),vocab_size*embedding_dim,12);
    cudaMalloc(&embedding_matrix_d,vocab_size*embedding_dim*sizeof(float));
    cudaMalloc(&tokens_d,batch_size*num_tokens*sizeof(int64_t));
    cudaMalloc(&output_embeds_d,batch_size*num_tokens*embedding_dim*sizeof(float));
    cudaMemcpy(tokens_d,tokens.data(),num_tokens*batch_size*sizeof(int64_t),cudaMemcpyDefault);
    cudaMemcpy(embedding_matrix_d,embedding_matrix.data(),vocab_size*embedding_dim*sizeof(float),cudaMemcpyDefault);
    GetEmbedding(embedding_matrix_d,tokens_d,output_embeds_d,num_tokens,batch_size,embedding_dim);
    cudaMemcpy(output_embeds.data(),output_embeds_d,batch_size*num_tokens*embedding_dim*sizeof(float),cudaMemcpyDefault);
    cudaFree(embedding_matrix_d);
    cudaFree(tokens_d);
    cudaFree(output_embeds_d);
    for (int i=0;i<batch_size*num_tokens;i++) {
        std::cout<<tokens[i]<<" ";
    }
    std::cout<<std::endl;
    for (int row=0;row<vocab_size;row++) {
        for (int col=0;col<embedding_dim;col++) {
            std::cout<<embedding_matrix[row*embedding_dim+col]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    std::cout<<"embedding start"<<std::endl;
    for (int b=0;b<batch_size;b++) {
        for (int row=0;row<num_tokens;row++) {
            for (int col=0;col<embedding_dim;col++) {
                std::cout<<output_embeds[col*1+row*embedding_dim+b*embedding_dim*num_tokens]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    fmt::print("hello");
    return 0;
}
