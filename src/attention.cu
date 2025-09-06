#include "attention.cuh"
#include <cuda_runtime.h>

// single head attention
// Q is Mxd, K is Nxd and V is Nxd
__global__ void QKTKernel(const float * q,const float* k,float* output,int m ,int n,int d)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int index=row*n+col;
    if(row<m && col<n)
    {
        float val=0.0f;
        for(int i=0;i<d;i++)
        {
            val+=q[row*d+i]*k[col*d+i];
        }
        val/=sqrtf(static_cast<float>(d));
        output[index]=val;
    }
}

__global__ void SoftmaxKernel(const float * QKt,float * output,int m,int n)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<m)
    {
        float row_max=-INFINITY;
        for(int col=0;col<n;col++)
        {
            row_max=fmaxf(row_max,QKt[row*n+col]);
        }
        float sum=0.0f;
        for(int col=0;col<n;col++)
        {
            float val=expf(QKt[row*n+col]-row_max);
            output[row*n+col]=val;
            sum+=val;
        }
        for(int col=0;col<n;col++)
        {
            output[row*n+col]/=sum;
        }
    }
}
__global__ void SVKernel(const float * S,const float* V,float* output,int m ,int n,int d)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int index=row*d+col;
    if(row<m && col<d)
    {
        float val=0.0f;
        for(int i=0;i<n;i++)
        {
            val+=S[row*n+i]*V[i*d+col];
        }
        output[index]=val;
    }
}

void GetAttention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float * QKT;
    float * softmaxOutputs;
    cudaMalloc(&QKT,M*N*sizeof(float));
    cudaMalloc(&softmaxOutputs,M*N*sizeof(float));
    dim3 threads1(16,16);
    dim3 blocks1((N+threads1.x-1)/threads1.x,(M+threads1.y-1)/threads1.y);
    QKTKernel<<<blocks1,threads1>>>(Q,K,QKT,M,N,d);
    cudaDeviceSynchronize();
    int threads2=256;
    int blocks2=(M+threads2-1)/threads2;
    SoftmaxKernel<<<blocks2,threads2>>>(QKT,softmaxOutputs,M,N);
    cudaDeviceSynchronize();
    dim3 threads3(16,16);
    dim3 blocks3((d+threads3.x-1)/threads3.x,(M+threads3.y-1)/threads3.y);
    SVKernel<<<blocks3,threads3>>>(softmaxOutputs,V,output,M,N,d);
    cudaDeviceSynchronize();
    cudaFree(QKT);
    cudaFree(softmaxOutputs);
}

// multihead attention
// Q,K,V all have size Nxd_model and h are the number of heads, and d_model is divisible by h,
// d_k = d_model/h
__global__ void HeadWiseQKT(const float* Q, const float* K, float* output, int N, int d_model, int h) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.z * blockDim.z + threadIdx.z;

    if (head < h && row < N && col < N) {
        int dk = d_model / h;
        float val = 0.0f;
        for (int i = 0; i < dk; i++) {
            float q = Q[row * d_model + head * dk + i];
            float k = K[col * d_model + head * dk + i];  // row-major access
            val += q * k;
        }
        output[head * N * N + row * N + col] = val / sqrtf((static_cast<float>(dk)));
    }
}

__global__ void HeadWiseSoftmax(const float* qkt, float* output, int N, int h) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;

    if (head < h && row < N) {
        float max_val = -INFINITY;
        for (int col = 0; col < N; col++) {
            max_val = fmaxf(max_val, qkt[head * N * N + row * N + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            float val = expf(qkt[head * N * N + row * N + col] - max_val);
            output[head * N * N + row * N + col] = val;
            sum += val;
        }

        for (int col = 0; col < N; col++) {
            output[head * N * N + row * N + col] /= sum;
        }
    }
}

__global__ void HeadWiseSV(const float* S, const float* V, float* output, int N, int d_model, int h) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.z * blockDim.z + threadIdx.z;

    if (int dk = d_model / h; head < h && row < N && col < dk) {
        float val = 0.0f;
        for (int i = 0; i < N; i++) {
            float s = S[head * N * N + row * N + i];
            float v = V[i * d_model + head * dk + col];
            val += s * v;
        }
        output[row * d_model + head * dk + col] = val;
    }
}

void GetMultiHeadAttention(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    int dk = d_model / h;
    float *qkt, *softmaxOutput;

    cudaMalloc(&qkt, h * N * N * sizeof(float));
    cudaMalloc(&softmaxOutput, h * N * N * sizeof(float));
    cudaMemset(output, 0, N * d_model * sizeof(float));

    dim3 threads1(8, 8, 8);
    dim3 blocks1((h + threads1.x - 1) / threads1.x,
                 (N + threads1.y - 1) / threads1.y,
                 (N + threads1.z - 1) / threads1.z);
    HeadWiseQKT<<<blocks1, threads1>>>(Q, K, qkt, N, d_model, h);
    cudaDeviceSynchronize();

    dim3 threads2(8, 8);
    dim3 blocks2((h + threads2.x - 1) / threads2.x,
                 (N + threads2.y - 1) / threads2.y);
    HeadWiseSoftmax<<<blocks2, threads2>>>(qkt, softmaxOutput, N, h);
    cudaDeviceSynchronize();

    dim3 threads3(8, 8, 8);
    dim3 blocks3((h + threads3.x - 1) / threads3.x,
                 (N + threads3.y - 1) / threads3.y,
                 (dk + threads3.z - 1) / threads3.z);
    HeadWiseSV<<<blocks3, threads3>>>(softmaxOutput, V, output, N, d_model, h);
    cudaDeviceSynchronize();

    cudaFree(qkt);
    cudaFree(softmaxOutput);
}
