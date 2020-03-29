#include<iostream>
#include<omp.h>
#include<cmath>

using namespace std;

__device__ float d_a[32],d_d[32];
__device__ float d_e[32],d_f[32];

#define NITERATIOS (1024*1024)

#ifdef ILP4
//指令级并行
#define OP_COUNT 4*2*NITERATIOS

__global__ void
kernel(float a, float b, float c){

  register float d=a, e=a, f=a;

#pragma unroll 16
  for(int i=0; i<NITERATIOS; i++){
    a = a*b + c;
    d = d*b + c;
    e = e*b + c;
    f = f*b + c;
  }
  //写入全局,防止被优化
  d_a[threadIdx.x] = a;
  d_d[threadIdx.x] = d;
  d_e[threadIdx.x] = e;
  d_f[threadIdx.x] = f;
  
}

#else
//测试线程级并行
#define OP_COUNT 1*2*NITERATIOS

__global__ void
kernel(float a, float b, float c){

#pragma unroll 16
  for(int i=0; i<NITERATIOS; i++){
    a = a*b+c;
  }
  //写入全局,防止被优化
  d_a[threadIdx.x] = a;
}
#endif

int
main(){
  for(int nThreads = 32; nThreads<=1024; nThreads += 32){
    double st = omp_get_wtime();
    kernel<<<1, nThreads>>>(1.0, 2.0, 3.0);
    if(cudaGetLastError() != cudaSuccess){
      cerr<<"Launch error"<<endl;
      return 1;
    }
    cudaDeviceSynchronize();
    double end = omp_get_wtime();
    cout<<"warps:"<<ceil(nThreads/32)<<" "
        <<nThreads<<" "<<(nThreads*(OP_COUNT/1.e9)/(end-st))
        <<" Gflops"<<endl;
  }
  return 0;
}

