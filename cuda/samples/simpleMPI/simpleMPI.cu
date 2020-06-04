#include<iostream>
#include"simpleMPI.h"
#include"helper_cuda.h"

using namespace std;



//特意把可以放cpp的放cu这边
float sum(float *data, int size){
  float acc = 0.f;
  for(int i = 0; i<size; i++){
    acc += data[i];
  }
  return acc;
}

__global__ void
MPIKernel(float *input, float *output){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  output[tid] = sqrt(input[tid]);
}

void initData(float *data, int dataSize){
  for(int i=0; i<dataSize; i++)
    data[i] = static_cast<float>(rand()) / RAND_MAX;
}

void computeGPU(float *hostData, int blockSize, int gridSize){
  int dataSize = blockSize*gridSize;
  
  // device mem
  float *deviceInputData = nullptr;
  checkCudaErrors(cudaMalloc((void**)&deviceInputData, dataSize*sizeof(float)));

  float *deviceOutputData = nullptr;
  checkCudaErrors(cudaMalloc((void**)&deviceOutputData, dataSize*sizeof(float)));

  //cp host to device
  checkCudaErrors(cudaMemcpy(deviceInputData, hostData, dataSize*sizeof(float),
                              cudaMemcpyHostToDevice));

  MPIKernel<<<gridSize, blockSize>>>(deviceInputData,deviceOutputData);

  checkCudaErrors(cudaMemcpy(hostData, deviceOutputData, dataSize*sizeof(float),
                            cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(deviceInputData));
  checkCudaErrors(cudaFree(deviceOutputData));
}
