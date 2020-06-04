#include<sys/utsname.h>
#include<iostream>
#include<cassert>
#include<cuda_runtime.h>
#include<helper_functions.h>
#include<helper_cuda.h>

using std::cout;
using std::endl;

__global__ void
testKernel(int N){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  assert(tid < N);
}


int
main(int argc, char *argv[]){
  cout<<argv[0]<<" starting..."<<endl;

  int devId;
  int Nblocks = 2;
  int Nthreads = 32;
  cudaError_t error_id;

  utsname OS_System_Type;
  uname(&OS_System_Type);

  cout<<"OS_System_Type.release = "<<OS_System_Type.release<<endl;
  cout<<"OS Info: <"<<OS_System_Type.version<<">"<<endl;
  cout<<"=================="<<endl;
 
//  char const ** tmp = argv;
  char const* tmp = *argv;
  char const** tmp2 = &tmp;
  devId = findCudaDevice(argc, tmp2 );
  cout<<"devID:"<<devId<<endl;
  
  dim3 grid(Nblocks);
  dim3 block(Nthreads);

  cout<<"Launch kernel to generate asseration failures"<<endl;  
  testKernel<<<grid,block>>>(60);

  cout<<" will run synchronize"<<endl;
  error_id = cudaDeviceSynchronize();
  
  if(error_id == cudaErrorAssert){
    cout<<"Device assert failed as expected, CUDA error message is:"
        <<cudaGetErrorString(error_id)<<endl;
    exit(EXIT_FAILURE);
  }
  exit(EXIT_SUCCESS);
  
  



}
