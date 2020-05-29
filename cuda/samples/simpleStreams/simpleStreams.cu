//这个在很多书籍中都没详细说明，比如cudaDeviceScheduleYield 
//这个就只存在cuda_runtime文档中,
#include<cassert>
#include<cstdio>
#include<iostream>
#include<helper_cuda.h>

#include<sys/mman.h>

using std::endl;
using std::cout;

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1)) &(~(size-1)) )


__global__ void
init_array(int *g_data, int *factor, int num_iter){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i=0; i<num_iter; i++){
    g_data[idx] += *factor; //意图消耗时间，都获取同一个值，*factor，从而读操作无法合并,造成争抢
  }
}


void 
check(int cuda_device, 
      bool &bPinGenericMem,
      float &scale_factor,
      int n){

  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop,cuda_device));

  //cout<<"sync_method for CPU/GPU synchronization"<<endl;
  if(bPinGenericMem)
    cout<<"use_generic_memory use generic page-aligned for system memory"<<endl;
  else
    cout<<"use_cuda_malloc_host use cudaMallocHost to allocate system memory"<<endl;
  // 检测GPU是否可以map host memory(generic method)
  if(bPinGenericMem){

    cout<<"Device: "<<prop.name<<" canMapHostMemory:"
        <<(prop.canMapHostMemory?"Yes":"No")<<endl;

    if(prop.canMapHostMemory == 0){
       cout<<"Using cudaMallocHost, CUDA device does not support mapping of generic host memory"<<endl;
       bPinGenericMem = false;
    }
  }

  scale_factor = max(1.0f, (32.0f/(_ConvertSMVer2Cores(prop.major, prop.minor) * \
                                   (float)prop.multiProcessorCount)));
  // rint: Rounds x to an integral value, 
  //using the rounding direction specified by fegetround.
  n = (int)rint((float)n / scale_factor);  
  cout<<"> CUDA Capable: SM "<<prop.major<<"."<<prop.minor<<endl;
  cout<<"> "<<prop.multiProcessorCount<<" Multiprocessor(s) x "
      <<_ConvertSMVer2Cores(prop.major, prop.minor)<<" (Cores/Multiprocessor) = "
      <<_ConvertSMVer2Cores(prop.major, prop.minor)*prop.multiProcessorCount<<" (Cores)"<<endl;
  
  cout<<"> scale_factor = "<<1.0f/scale_factor<<endl;
  cout<<"> array_size = "<<n<<endl;
}


void 
allocateHostMemory(bool bPinGenericMem, int **pp_a, 
                   int **ppAligned_a, int nbytes){
#if CUDART_VERSION >= 4000
  if(bPinGenericMem){
    cout<<"> mmap() allocating "<<(float)nbytes/1048576.0f
        <<" Mbytes (generic page-aligned system memory)"<<endl;
    cout<<"调用linux的内存分配"<<endl;
    *pp_a = (int *)mmap(NULL,
                        (nbytes+MEMORY_ALIGNMENT), PROT_READ|PROT_WRITE,
                        MAP_PRIVATE|MAP_ANON, -1, 0);

    *ppAligned_a = (int *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);
    cout<<"将linux方式申请的内存，注册到cuda体系中"<<endl;
    cout<<"> cudaHostRegister() registering "
        <<(float)nbytes/1048576.0f<<" Mbytes of generic allocated system memory"<<endl;
    checkCudaErrors(cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped));
  }else
#endif
  {
    cout<<"> cudaMallocHost() allocating "<<(float)nbytes/1048576.0f
        <<" Mbytes of system memory"<<endl;
    checkCudaErrors(cudaMallocHost((void**)pp_a, nbytes));
    *ppAligned_a = *pp_a;
  }
}

int
main(int argc, char *argv[]){
  int nstreams = 4;
  int nreps=10;
  int n = 16*1024*1024;
  int nbytes = n*sizeof(int);
  dim3 threads,blocks;
  float epalsed_time, time_memcpy, time_kernel;
  float scale_factor = 1.0f;

  int device_sync_method = cudaDeviceBlockingSync; //默认同步方式
  int niters;

  
  int cuda_device=0;
  checkCudaErrors(cudaSetDevice(cuda_device));

  //就是page-locked memory是用linux的接口mmap 还是cuda封装的接口cudaMallocHost
  bool bPinGenericMem = true;
  check(cuda_device, bPinGenericMem, scale_factor, n);
  
  // 分配host 内存
  int c = 5; // 数组初始化的值
  int *h_a = nullptr;
  int *hAligned_a = nullptr; //指向已经对齐到MEMORY_ALIGNMENT的host mem

//  allocateHostMemory(bPinGenericMem, &h_a, &hAligned_a, nbytes);
  
  
  
  

  
}
