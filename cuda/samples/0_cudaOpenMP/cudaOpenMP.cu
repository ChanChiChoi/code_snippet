// Multi-GPU sample using OpenMP for threading on the CPU side
// needs a compiler that supports OpenMP 2.0

//stdio functions are used since C++ streams aren't necessarily thread safe
//#include<iostream>
#include<cstdio>
#include<memory>
#include<omp.h>

#include<helper_cuda.h>

typedef void(*fp)(int*);

__global__ void
kernel(int *g_data, int const b){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  g_data[tid] += b;
}

int
check(std::unique_ptr<int[]>&data, int const n, int const b){
  for(int i=0; i<n; i++)
    if(data[i] != i+b)
      return 0;
  return 1;
}


int 
main(int argc, char *argv[]){
  int num_gpus = 0;
  printf("%s Starting...\n\n",argv[0]);

  cudaGetDeviceCount(&num_gpus);
  if(num_gpus<1){
    printf("no CUDA capable deive swere detected\n");
    exit(EXIT_FAILURE);
  }

  //显示cpu 和gpu的数目
  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("number of CUDA devices:\t%d\n", num_gpus);

  for(int i=0;i<num_gpus; i++){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,i);
    printf("  %d: %s\n",i,prop.name);
  }
  printf("---------------------------------\n");


  unsigned int n = num_gpus*8192;
  unsigned int nbytes = n*sizeof(int);
  int val = 3;
  std::unique_ptr<int[]>h_data{new int[n]};

  for(unsigned int i = 0;i<n;i++)
    h_data[i] = i;

////////////////////////////////////////////////////////////////
// run as many CPU threads as there are CUDA devices
//   each CPU thread controls a different device, processing its
//   portion of the data.  It's possible to use more CPU threads
//   than there are CUDA devices, in which case several CPU
//   threads will be allocating resources and launching kernels
//   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
//   Recall that all variables declared inside an "omp parallel" scope are
//   local to each CPU thread
//
  //创建num_gpus个线程
  omp_set_num_threads(num_gpus);  
  //调用omp特殊宏
  #pragma omp parallel
  {
    unsigned int cpu_thread_id = omp_get_thread_num();//获取当前线程id
    unsigned int num_cpu_threads = omp_get_num_threads();//一共开启的线程数
    //为线程指定GPU
    int gpu_id = -1;
    checkCudaErrors(cudaSetDevice(cpu_thread_id%num_gpus)); 
    checkCudaErrors(cudaGetDevice(&gpu_id));
    printf("thread id:%d (of %d) uses GPU %d\n",cpu_thread_id,num_cpu_threads,gpu_id);
 
    //每个线程的数据地址和字节大小
    int *sub_hData = h_data.get() + cpu_thread_id*n/num_cpu_threads;
    unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;

    dim3 blocks(128);
    dim3 grid(n/(blocks.x*num_cpu_threads));

    int *_d_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&_d_data, nbytes_per_kernel));
    std::unique_ptr<int,fp>d_data{_d_data,[](int*p){cudaFree(p);}};

    checkCudaErrors(cudaMemset(d_data.get(), 0, nbytes_per_kernel));
    checkCudaErrors(cudaMemcpy(d_data.get(), sub_hData, nbytes_per_kernel,
                               cudaMemcpyHostToDevice));

    kernel<<<grid, blocks>>>(d_data.get(),val);

    checkCudaErrors(cudaMemcpy(sub_hData, d_data.get(), nbytes_per_kernel,
                               cudaMemcpyDeviceToHost));
  }
  printf("---------------------------\n");

  if(cudaSuccess != cudaGetLastError())
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
  exit(check(h_data, n, val)? EXIT_SUCCESS:EXIT_FAILURE);


}


