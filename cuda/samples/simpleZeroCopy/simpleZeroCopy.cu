#include<cstdio>
#include<cassert>

#include<helper_cuda.h>
#include<dynlink/cuda_drvapi_dynlink_cuda.h>
#include<iostream>


#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) ( ((size_t)x+(size-1)) & (~(size-1)) )

using std::cout;
using std::endl;

__global__ void
vectorAddGPU(float *a, float *b, float *c, int N){
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid<N)
    c[tid] = a[tid] + b[tid];
}

int
main(int argc, char *argv[]){
  
  bool bPinGenericMemory = false;
  int idev = 0;
  float *a, *b, *c;             // Pinned memory allocated on the host
  float *a_UA, *b_UA, *c_UA;    // Non-4K Aligned Pinned memry on the host
  float *d_a, *d_b, *d_c;
  
  // GPU SM 必须大于1.2 
  bPinGenericMemory = true;
  if(bPinGenericMemory)
    cout<<"> Using Generic System Paged Memory (malloc)"<<endl;
  else
    cout<<"> Using CUDA Host Allocated (cudaHostAlloc)"<<endl;

  checkCudaErrors(cudaSetDevice(idev));
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, idev));
#if CUDART_VERSION >= 2020
  if(!prop.canMapHostMemory){
    cout<<"Device "<<idev<<" does not support mapping CPU host memory"<<endl;
    exit(EXIT_FAILURE);
  }
  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
#else
  cout<<"CUDART version "<<CUDART_VERSION/1000<<"."<<(CUDART_VERSION%100)/10
      <<" does not support <cudaDeviceProp.canMapHostMemory> field"<<endl;
  exit(EXIT_FAILURE);
#endif

  //分配mapped CPU mem
  int nelem = 1048576;
  int bytes = nelem*sizeof(float);  

  if(bPinGenericMemory){
#if CUDART_VERSION >= 4000
    a_UA = (float *)malloc(bytes+MEMORY_ALIGNMENT);
    b_UA = (float *)malloc(bytes+MEMORY_ALIGNMENT);
    c_UA = (float *)malloc(bytes+MEMORY_ALIGNMENT);
    // 需要确保内存是对齐到4K的
    a = (float *)ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
    b = (float *)ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
    c = (float *)ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

    checkCudaErrors(cudaHostRegister(a, bytes, CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostRegister(b, bytes, CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostRegister(c, bytes, CU_MEMHOSTALLOC_DEVICEMAP));
#endif
   }else{
#if CUDART_VERSION >= 2020
    unsigned int flags;
    flags = cudaHostAllocMapped;
    checkCudaErrors(cudaHostAlloc((void**)&a, bytes, flags));
    checkCudaErrors(cudaHostAlloc((void**)&b, bytes, flags));
    checkCudaErrors(cudaHostAlloc((void**)&c, bytes, flags));
#endif
   }

   //初始化
   for(int n=0; n<nelem; n++){
      a[n] = rand()/(float)RAND_MAX;
      b[n] = rand()/(float)RAND_MAX;
   }
   // 获取GPU侧指向pinned的指针
#ifdef WITHOUT_UVA
  cout<<"USE cudaHostGetDevicePointer"<<endl;
  #if CUDART_VERSION >= 2020
     checkCudaErrors(cudaHostGetDevicePointer((void**)&d_a, (void*)a,0));
     checkCudaErrors(cudaHostGetDevicePointer((void**)&d_b, (void*)b,0));
     checkCudaErrors(cudaHostGetDevicePointer((void**)&d_c, (void*)c,0));
  #endif
#else
   cout<<"DONOT USE cudaHostGetDevicePointer"<<endl;
   d_a = a; d_b = b; d_c = c;
#endif

  
   cout<<"> vectorAddGPU kernel will add vectors using mapped CPU memory"<<endl;
   dim3 block(256);
   dim3 grid((unsigned int)ceil(nelem/(float)block.x));
   vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem); 
   checkCudaErrors(cudaDeviceSynchronize());
   getLastCudaError("vectorAddGPU() execution failed\n");

   float errorNorm = 0.f;
   float ref,diff,refNorm;
   for(int n=0; n<nelem; n++){
      ref = a[n]+b[n];
      diff = c[n] - ref;
      errorNorm += diff*diff;
      refNorm += ref*ref;
   }
   cout<<"errorNorm:"<<(float)sqrt((double)errorNorm)<<endl;
   cout<<"refNorm:"<<(float)sqrt((double)refNorm)<<endl;
   if(errorNorm/refNorm < 1.e-6f)
     cout<<"PASSED"<<endl;
    else
     cout<<"FAILED"<<endl;

   cout<<"> Releasing CPU memory..."<<endl;
   if(bPinGenericMemory){
#if CUDART_VERSION >= 4000
     checkCudaErrors(cudaHostUnregister(a));
     checkCudaErrors(cudaHostUnregister(b));
     checkCudaErrors(cudaHostUnregister(c));
     free(a_UA);
     free(b_UA);
     free(c_UA);
#endif
   }else{
#if CUDART_VERSION >= 2020
     checkCudaErrors(cudaFreeHost(a));
     checkCudaErrors(cudaFreeHost(b));
     checkCudaErrors(cudaFreeHost(c));
#endif
   }

}
