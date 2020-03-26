#include<iostream>
#include<cassert>

#include<cuda_runtime.h>
#include<helper_cuda.h>

using std::cout;
using std::endl;

#define NBLOCKS 64
#define NTHREADS 256

__global__ static void
timeReduction(float const* input, float *output, clock_t* timer){

  // __shared__ float shared[2*blockDim.x];
  extern __shared__ float shared[];
  int const tid = threadIdx.x;
  int const bid = blockIdx.x;

  if(tid==0) 
    timer[bid] = clock();

  shared[tid] = input[tid];
  shared[tid + blockDim.x] = input[tid + blockDim.x];

  //perform reduction to find minimum
  for(int d = blockDim.x; d>0; d/=2){
    __syncthreads();

    if(tid<d){
      float f0 = shared[tid];
      float f1 = shared[tid+d];

      if(f1<f0){
        shared[tid] = f1;
      }
    }
  }

  if (tid == 0)
    output[bid] = shared[0];
  __syncthreads();

  if(tid == 0)
   timer[bid+gridDim.x] = clock();

}



int
main(int argc, char *argv[]){

  cout<<"CUDA Clock sample"<<endl;
  char const* tmp1 = *argv;
  char const** tmp2 = &tmp1;
  int dev = findCudaDevice(argc,tmp2);

  float *dinput = nullptr;
  float *doutput = nullptr;
  clock_t *dtimer = nullptr;

  clock_t timer[NBLOCKS*2];
  float input[NTHREADS*2];

  for(int i=0;i<NTHREADS; i++)
    input[i] = static_cast<float>(i);

  checkCudaErrors(cudaMalloc((void **)&dinput, sizeof(float)*NTHREADS*2));  
  checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float)*NBLOCKS));  
  checkCudaErrors(cudaMalloc((void **)&dtimer, sizeof(clock_t)*NBLOCKS*2));  

  checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float)*NTHREADS*2, cudaMemcpyHostToDevice));

  timeReduction<<<NBLOCKS,NTHREADS,sizeof(float)*2*NTHREADS>>>(dinput,doutput,dtimer);

  checkCudaErrors(cudaMemcpy(timer, dtimer,sizeof(clock_t)*NBLOCKS*2, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(dinput));
  checkCudaErrors(cudaFree(doutput));
  checkCudaErrors(cudaFree(dtimer));

  long double avgElapsedClocks = 0;
  for(int i=0; i<NBLOCKS; i++){
    avgElapsedClocks+=static_cast<long double>(timer[i+NBLOCKS]-timer[i]);
  }

  avgElapsedClocks = avgElapsedClocks/NBLOCKS;
  cout<<"average clocks/block = "<<avgElapsedClocks<<endl;
  
  exit(EXIT_SUCCESS);

}
