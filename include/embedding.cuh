#ifndef EMBEDDING_CUH
#define EMBEDDING_CUH

void GetEmbedding(const float * EmbeddingWieghts,const int64_t* tokens,float * output,int64_t n,int64_t batch_size,int64_t embedding_dim);

#endif
