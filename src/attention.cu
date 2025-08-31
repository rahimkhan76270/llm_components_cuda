

// single head attention
// Q is Mxd, K is Nxd and V is Nxd
#include <math.h>
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
        val/=sqrtf((float)d);
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
    float * sftmax;
    cudaMalloc(&QKT,M*N*sizeof(float));
    cudaMalloc(&sftmax,M*N*sizeof(float));
    dim3 threads1(16,16);
    dim3 blocks1((N+threads1.x-1)/threads1.x,(M+threads1.y-1)/threads1.y);
    QKTKernel<<<blocks1,threads1>>>(Q,K,QKT,M,N,d);
    cudaDeviceSynchronize();
    int threads2=256;
    int blocks2=(M+threads2-1)/threads2;
    SoftmaxKernel<<<blocks2,threads2>>>(QKT,sftmax,M,N);
    cudaDeviceSynchronize();
    dim3 threads3(16,16);
    dim3 blocks3((d+threads3.x-1)/threads3.x,(M+threads3.y-1)/threads3.y);
    SVKernel<<<blocks3,threads3>>>(sftmax,V,output,M,N,d);
    cudaDeviceSynchronize();
    cudaFree(QKT);
    cudaFree(sftmax);
}