#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<iostream>
#include<memory>

using namespace std;

#include<helper_cuda.h>

//包含kernel头文件
#include "simpleAtomicIntrinsics_kernel.cuh"

extern "C" bool
computeGold(int *gpuData, int const len);

int
main(int argc, char *argv[]){
  cout<<argv[0]<<" Starting..."<<endl;
  

  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 11;
  unsigned int memSize = sizeof(int)*numData;

  // host
  unique_ptr<int[]>h_output{new int[numData]};
  for(unsigned int i=0; i<numData; i++)
     h_output[i] = 0;
  h_output[8] = h_output[10] = 0xFF;//随便赋值，不要让h_output全都是0

  //display
  for(unsigned int i=0; i<numData; i++)
    cout<<i<<":"<<h_output[i]<<endl;

  // device
  int *_d_output;
  checkCudaErrors(cudaMalloc((void**)&_d_output, memSize));
  unique_ptr<int, void(*)(int*)> d_output{_d_output, [](int*p){cudaFree(p);}};
  checkCudaErrors(cudaMemcpy(d_output.get(), h_output.get(), memSize,
                             cudaMemcpyHostToDevice));

  // kernel
  kernel<<<numBlocks, numThreads>>>(d_output.get());
  getLastCudaError(" kernel execution failed");
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_output.get(), d_output.get(), memSize,
                             cudaMemcpyDeviceToHost));

  //check
  computeGold(h_output.get(), numThreads*numBlocks);

  cout<<"----------------"<<endl;
  for(unsigned int i=0; i<numData; i++)
    cout<<i<<":"<<h_output[i]<<endl;

  exit(EXIT_SUCCESS);

}
