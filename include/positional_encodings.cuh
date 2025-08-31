#ifndef POSITIONAL_ENCODINGS_CUH
#define POSITIONAL_ENCODINGS_CUH

void AddAbsolutePositionalEncodings(float *input,int64_t num_tokens,int64_t embedding_dim,int64_t d_model,int64_t batch_size);

#endif
