#include<iostream>
#include<helper_cuda.h>

using std::cout;
using std::endl;
using std::cerr;

constexpr int manualBlockSize = 32;

__global__ void
square(int *array, int arrayCount){

  extern __shared__ int dynamicSMem[];
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx < arrayCount)
    array[idx] *= array[idx];
}


double
reportPotentialOccupancy(void *kernel, int block, size_t dynamicSMem){
  int device;
  cudaDeviceProp prop;
  int nBlock;
  int activeWarps;
  int maxWarps;
  double occupancy;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop,device));
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &nBlock,
                      kernel,
                      block,
                      dynamicSMem
                     ));

  activeWarps = nBlock*block / prop.warpSize;
  maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  occupancy = (double)activeWarps / maxWarps;
  return occupancy;
}

int
launchConfig(int *array, int arrayCount, bool automatic){
  int block;
  int minGrid;
  int grid;
  size_t dynamicSMemUsage = 0;
  
  cudaEvent_t st;
  cudaEvent_t ed;
  float elapsedTime;
  double potentialOccupancy;
  checkCudaErrors(cudaEventCreate(&st));
  checkCudaErrors(cudaEventCreate(&ed));

  if(automatic){
    //Returns grid and block size that achieves maximum potential occupancy for a device function.
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                     &minGrid,
                     &block,
                     (void*)square,
                     dynamicSMemUsage,
                     arrayCount
                    ));
    cout<<"suggested block size: "<<block<<endl;
    cout<<"minimum grid size for maximum occupancy: "<<minGrid<<endl;
    cout<<"dynamic mem:"<<dynamicSMemUsage<<endl;
  }else{
    block = manualBlockSize;
  }
  grid = (arrayCount+ block-1)/block;
  checkCudaErrors(cudaEventRecord(st));
  square<<<grid,block,dynamicSMemUsage>>>(array, arrayCount);
  checkCudaErrors(cudaEventRecord(ed));
  checkCudaErrors(cudaDeviceSynchronize());

  potentialOccupancy = reportPotentialOccupancy((void*)square, block,dynamicSMemUsage);
  cout<<"Potential occupancy: "<<potentialOccupancy*100<<"%"<<endl;
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, st, ed));
  cout<<"Elapsed time: "<<elapsedTime<<" ms"<<endl;
  return 0;
  
  
}

int
test(bool automaticLaunchConfig, int const count = 1000000){
  int *array;
  int *dArray;
  int size = count*sizeof(int);
  array = new int[count];

  for(int i=0; i<count; i++)
    array[i] = i;

  checkCudaErrors(cudaMalloc(&dArray, size));
  checkCudaErrors(cudaMemcpy(dArray,array,size,cudaMemcpyHostToDevice));
  for(int i=0; i<count; i++)
    array[i]=0;

  launchConfig(dArray, count, automaticLaunchConfig);

  checkCudaErrors(cudaMemcpy(array, dArray, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(dArray));

  for(int i=0; i<count; i++)
    if(array[i] != i*i){
      cout<<"element:"<<i<<" expected:"<<i*i<<" actual:"<<array[i]<<endl;
      return 1;
    }

  delete [] array;
  return 0;
}


int
main(){
  int status;
  //----------
  cout<<"starting simple occupancy"<<endl<<endl;
  cout<<"[ manual configuration with "<<manualBlockSize
      <<" threads per block ]"<<endl;
  status = test(false);
  if(status){
    cerr<<"Test Failed"<<endl;
    return -1;
  }
  //-----------
  cout<<endl;
  cout<<"[ Automic, occupancy-based configuration ]"<<endl;
  status = test(true);
  if(status){
    cerr<<"Test Failed"<<endl;
    return -1;
  }
  //----------
  cout<<endl;
  cout<<"Test PASSED"<<endl;
  return 0;
  
}
