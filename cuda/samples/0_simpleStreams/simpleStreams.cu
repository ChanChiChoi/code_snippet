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

 char const*  sDeviceSyncMethod[] = {
 "cudaDeviceScheduleAuto",
 "cudaDeviceScheduleSpin",
 "cudaDeviceScheduleYield",
 "INVALID",
 "cudaDeviceScheduleBlockingSync",
 NULL
};

void 
check(int cuda_device, 
      bool &bPinGenericMem,
      float &scale_factor,
      int n,
      int device_sync_method){

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

  //-------------------??????
  cout<<"> Using CPU/GPU Device Synchronization method:"<<sDeviceSyncMethod[device_sync_method]<<endl;
  checkCudaErrors(cudaSetDeviceFlags(device_sync_method |\
                                     (bPinGenericMem ? cudaDeviceMapHost : 0) 
                                     ));
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

bool
correct_data(int *a, int const n, int const c){
  for(int i=0; i<n; i++){
    if(a[i] != c){
     cout<<i<<", "<<a[i]<<" "<<c<<endl;
     exit(EXIT_FAILURE);
    }
  }
  return true;
}

inline void
freeHostMemory(bool bPinGenericMem, int **pp_a, int **ppAligned_a, int nbytes){
#if CUDART_VERSION >= 4000
  if(bPinGenericMem){
    checkCudaErrors(cudaHostUnregister(*ppAligned_a));
    munmap(*pp_a, nbytes);
  }
  else
#endif
  {
   cudaFreeHost(*pp_a);
  }

}

int
main(int argc, char *argv[]){
  int nstreams = 4;
  int nreps=10;
  int n = 16*1024*1024;
  int nbytes = n*sizeof(int);
  dim3 threads,blocks;
  float elapsed_time, time_memcpy, time_kernel;
  float scale_factor = 1.0f;

  //本程序中，device_sync_method可选0 1 2 3 4,其中3 为非法，对应sDeviceSyncMethod中选项
  int device_sync_method = cudaDeviceBlockingSync; //默认同步方式为4
  //device_sync_method = 2;
  int niters=5;
  
  int cuda_device=0;
  checkCudaErrors(cudaSetDevice(cuda_device));

  //就是page-locked memory是用linux的接口mmap 还是cuda封装的接口cudaMallocHost
  bool bPinGenericMem = true;
  check(cuda_device, bPinGenericMem, scale_factor, n, device_sync_method);
  
  // 分配host 内存
  int c = 5; // 数组初始化的值
  int *h_a = nullptr;
  int *hAligned_a = nullptr; //指向已经对齐到MEMORY_ALIGNMENT的host mem

  allocateHostMemory(bPinGenericMem, &h_a, &hAligned_a, nbytes);

  // 分配device 内存，然后初始化
  int *d_a = nullptr;
  int *d_c = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 0x0, nbytes));
  checkCudaErrors(cudaMalloc((void**)&d_c, sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice));

  cout<<"Starting Test..."<<endl;
  
  //分配和初始化stream handles的数组
  cudaStream_t *streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
  for(int i=0; i<nstreams; i++){
    checkCudaErrors(cudaStreamCreate(&(streams[i])));
  }
  
  //创建cuda event; 使用blocking sync
  cudaEvent_t st, ed;
  int eventFlags = (device_sync_method == cudaDeviceBlockingSync) ? \
                   cudaEventBlockingSync : cudaEventDefault;
  checkCudaErrors(cudaEventCreateWithFlags(&st, eventFlags));
  checkCudaErrors(cudaEventCreateWithFlags(&ed, eventFlags));
  
  //计算device 到 host的memcpy时间
  checkCudaErrors(cudaEventRecord(st,0));// 放置在第0个流中，以此保证之前的cuda call都完成了
  checkCudaErrors(cudaMemcpyAsync(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost, streams[0]));
  checkCudaErrors(cudaEventRecord(ed,0));
  checkCudaErrors(cudaEventSynchronize(ed));
  checkCudaErrors(cudaEventElapsedTime(&time_memcpy, st, ed));
  cout<<"memcopy: "<<time_memcpy<<" ms"<<endl;

  //计算kernel的时间
  threads = dim3(512);
  blocks = dim3(n/threads.x);
  checkCudaErrors(cudaEventRecord(st,0));
  init_array<<<blocks, threads, 0, streams[0]>>>(d_a, d_c, niters);
  checkCudaErrors(cudaEventRecord(ed,0));
  checkCudaErrors(cudaEventSynchronize(ed));
  checkCudaErrors(cudaEventElapsedTime(&time_kernel, st, ed));
  cout<<"kernel: "<<time_kernel<<" ms"<<endl;
  
//===========================================
  threads = dim3(512);
  blocks = dim3(n/threads.x);
  checkCudaErrors(cudaEventRecord(st,0));
  for(int k=0; k<nreps; k++){
    init_array<<<blocks, threads>>>(d_a, d_c, niters);
    checkCudaErrors(cudaMemcpy(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost));
  }
  checkCudaErrors(cudaEventRecord(ed,0));
  checkCudaErrors(cudaEventSynchronize(ed));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, st, ed));
  cout<<"不采用多流方式: "<<elapsed_time<<" ms"<<endl;

//==========================================  
  threads = dim3(512);
  blocks = dim3(n/(nstreams*threads.x) );
  memset(hAligned_a, 255, nbytes);//将host的内存都设为1，为了后续结果正确性测试
  checkCudaErrors(cudaMemset(d_a, 0, nbytes));//将device内存设为0，为了后续测试
  checkCudaErrors(cudaEventRecord(st,0));
  for(int k=0; k<nreps; k++){
    //将流和内存复制分开，保证流都是异步执行的，
    for(int i=0; i<nstreams; i++){
      init_array<<<blocks, threads, 0, streams[i]>>>(d_a+i*n/nstreams, d_c, niters);
    }
    //异步执行nstreams 的内存复制，要注意的是，
    // 在第n个流中的内存复制是发送在当前第n个流中 之前的cuda call都完成的情况下
    for(int i=0; i<nstreams; i++)
      checkCudaErrors(cudaMemcpyAsync(hAligned_a+i*n/nstreams, d_a+i*n/nstreams,
                                      nbytes/nstreams, cudaMemcpyDeviceToHost,streams[i]));
  } 
  checkCudaErrors(cudaEventRecord(ed,0));
  checkCudaErrors(cudaEventSynchronize(ed));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, st, ed));
  cout<<"共 "<<nstreams<<" streams: "<<elapsed_time/nreps<<" ms"<<endl;

  //----------------
  cout<<"--------------------"<<endl;
  correct_data(hAligned_a, n, c*nreps*niters);  
  
  //free resources
  for(int i=0; i<nstreams; i++)
    checkCudaErrors(cudaStreamDestroy(streams[i]));

  checkCudaErrors(cudaEventDestroy(st));
  checkCudaErrors(cudaEventDestroy(ed));
  
  freeHostMemory(bPinGenericMem, &h_a, &hAligned_a, nbytes);

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_c));

  cout<<"PASSED !"<<endl;
  
}
