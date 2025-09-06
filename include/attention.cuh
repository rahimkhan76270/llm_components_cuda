#ifndef ATTENTION_CUH
#define ATTENTION_CUH
void GetAttention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) ;
void GetMultiHeadAttention(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h);
#endif
