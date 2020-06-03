// 这个例子就是个参照形式，没什么内容和知识点
#include<cstdlib>
#include<cstdio>
#include<cstring>
#include<cmath>
#include<iostream>

#include<helper_cuda.h>

using std::cout;
using std::endl;

extern "C"
void computeGold(float *ref, float *idata, unsigned int const len);

__global__ void
testKernel(float *g_idata, float *g_odata){

  extern __shared__ float sdata[];
  unsigned int const tid = threadIdx.x;
  unsigned int const num_threads = blockDim.x;

  sdata[tid] = g_idata[tid];
  __syncthreads();

  sdata[tid] = (float)num_threads *sdata[tid];
  __syncthreads();

  g_odata[tid] = sdata[tid];
}

void
runTest(int argc, char *argv[]){
  unsigned int num_threads=32;
  unsigned int mem_size = sizeof(float)*num_threads;

  float *h_idata = (float*)malloc(mem_size);
  for(unsigned int i=0; i<num_threads; i++)
    h_idata[i] = (float)i;

  float *d_idata;
  checkCudaErrors(cudaMalloc((void**)&d_idata, mem_size));
  checkCudaErrors(cudaMemcpy(d_idata, h_idata,mem_size,
                             cudaMemcpyHostToDevice));

  float *d_odata;
  checkCudaErrors(cudaMalloc((void**)&d_odata, mem_size));

  dim3 grid(1,1,1);
  dim3 threads(num_threads, 1,1);

  testKernel<<<grid, threads, mem_size>>>(d_idata, d_odata);
  getLastCudaError("Kernel execution failed");

  float *h_odata = (float*)malloc(mem_size);
  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float)*num_threads,
                             cudaMemcpyDeviceToHost));

  float *ref = (float*)malloc(mem_size);
  computeGold(ref, h_idata, num_threads);


  free(h_idata);
  free(h_odata);
  free(ref);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));
  cout<<"Shutdown..."<<endl;
}


int
main(int argc, char *argv[]){
  runTest(argc, argv);
}
