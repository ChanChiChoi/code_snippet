#include<iostream>
#include<helper_cuda.h>
#include<cuda_runtime.h>

//compileFileToPTX;loadPTX 都在nvrtc_helper
#include<nvrtc_helper.h>


// It's interesting to change the number of blocks and the number of threads to
// understand how to keep the hardware busy.

#define NUM_BLOCKS 64
#define NUM_THREADS 256
using namespace std;

int main(int argc, char *argv[]){
  cout<<"CUDA Clock sample"<<endl;

  typedef long clock_t;
  clock_t timer[NUM_BLOCKS*2];
  float input[NUM_THREADS*2];

  //初始化 input
  for(int i=0;i<NUM_THREADS*2;i++){
    input[i] = static_cast<float>(i);
  }
  char *ptx;
  char const *kernel_file;
  size_t ptxSize;

  //文件直接编译成ptx汇编代码，并存放在ptx地址处
  kernel_file = "clock_kernel.cu";
  compileFileToPTX(const_cast<char*>(kernel_file), argc, argv, &ptx, &ptxSize, 0); 

  //装载ptx
  CUmodule module = loadPTX(ptx,argc,argv);
  CUfunction kernel_addr;
  //获取global 函数
  checkCudaErrors(cuModuleGetFunction(&kernel_addr, 
                                      module, "timedReduction"));

  dim3 grid(NUM_BLOCKS);
  dim3 block(NUM_THREADS);

  CUdeviceptr dinput, doutput, dtimer;
  checkCudaErrors(cuMemAlloc(&dinput, sizeof(float)*NUM_THREADS*2));
  checkCudaErrors(cuMemAlloc(&doutput, sizeof(float)*NUM_BLOCKS));
  checkCudaErrors(cuMemAlloc(&dtimer, sizeof(clock_t)*NUM_BLOCKS*2));
  checkCudaErrors(cuMemcpyHtoD(dinput, input, sizeof(float)*NUM_THREADS*2));

  void *arr[] = {
         (void*)&dinput,
         (void *)&doutput,
         (void *)&dtimer
       };

  //runtimer 的接口
  checkCudaErrors(cuLaunchKernel(kernel_addr,
                                 grid.x,grid.y,grid.z,
                                 block.x, block.y, block.z,
                                 sizeof(float)*2*NUM_THREADS, 0, &arr[0], 0));

  checkCudaErrors(cuCtxSynchronize());
  checkCudaErrors(cuMemcpyDtoH(timer,dtimer,sizeof(clock_t)*NUM_BLOCKS*2)); 
  checkCudaErrors(cuMemFree(dinput));
  checkCudaErrors(cuMemFree(doutput));
  checkCudaErrors(cuMemFree(dtimer));

  long double avgElapsedClocks = 0;
  for(int i=0; i<NUM_BLOCKS; i++){
    avgElapsedClocks += static_cast<long double>(timer[i+NUM_BLOCKS]-timer[i]);
  }
  avgElapsedClocks = avgElapsedClocks/NUM_BLOCKS;
  cout<<"Average clocks/block = "<<avgElapsedClocks<<endl;
  exit(EXIT_SUCCESS);

}
