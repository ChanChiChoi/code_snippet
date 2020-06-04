#include<iostream>
#include<vector>
#include<cuda_runtime.h>
#include<helper_cuda.h>

#include "simpleDeviceLibrary.cuh"

using std::cout;
using std::endl;
using std::vector;

#define EPS 1e-5

typedef unsigned int uint;
using deviceFunc = float(*)(float);

__device__ deviceFunc
dMultiplyByTwoPtr = multiplyByTwo;

__device__ deviceFunc
dDivideByTwoPtr = divideByTwo;

__global__ void
transformVector(float *v, deviceFunc f, uint size){
  uint tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid<size){
    v[tid] = (*f)(v[tid]);
  }
}


void
test(int argc, char const**argv){

  uint const kVectorSize = 1000;
  vector<float>hVector(kVectorSize);
  for(uint i=0; i<kVectorSize; i++){
    hVector[i] = rand()/static_cast<float>(RAND_MAX);
  }

  float *dVector;
  checkCudaErrors(cudaMalloc(&dVector, kVectorSize*sizeof(float)));
  checkCudaErrors(cudaMemcpy(dVector, &hVector[0], kVectorSize*sizeof(float),
                             cudaMemcpyHostToDevice));

  int const nThreads = 1024;
  int const nBlocks = 1;
  dim3 grid(nBlocks);
  dim3 block(nThreads);

  // important
  // 函数对象复制
  deviceFunc hFunctionPtr;
  cudaMemcpyFromSymbol(&hFunctionPtr,dMultiplyByTwoPtr,sizeof(deviceFunc));
  transformVector<<<grid, block>>>(dVector, hFunctionPtr, kVectorSize);
  checkCudaErrors(cudaGetLastError());

  cudaMemcpyFromSymbol(&hFunctionPtr,dDivideByTwoPtr,sizeof(deviceFunc));
  transformVector<<<grid, block>>>(dVector, hFunctionPtr, kVectorSize);
  checkCudaErrors(cudaGetLastError());

  //results
  vector<float>hResultVector(kVectorSize);
  checkCudaErrors(cudaMemcpy(&hResultVector[0], dVector, kVectorSize*sizeof(float),
                             cudaMemcpyDeviceToHost));

  for(uint i=0; i<kVectorSize; i++){

    if( fabs(hVector[i] - hResultVector[i]) > EPS ) {
      cout<<"Computations were incorrect..."<<endl;
      exit(EXIT_FAILURE);
    }
  }

  if(dVector)
    checkCudaErrors(cudaFree(dVector));
}

int
main(int argc, char* argv[]){
  cout<<argv[0]<<" starting..."<<endl;
  try{
    test(argc, (char const **)argv);
  }catch(...){
    cout<<"Error occured, exiting..."<<endl;
    exit(EXIT_FAILURE);
  }
  cout<<"PASS !"<<endl;
}
