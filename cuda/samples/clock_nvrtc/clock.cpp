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
  char *ptx, *kernel_file;
  size_t ptxSize;

  //文件直接编译成ptx汇编代码，并存放在ptx地址处
  kernel_file = "clock_kernel.cu"
  compileFileToPTX(kernel_file, argc, argv, &ptx, &ptxSize, 0); 

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
  checkCudaErrors(cuMemAlloc(&dtimer, sizeof(float)*NUM_BLOCKS*2));
  checkCudaErrors(cuMemcyHtoD(dinput, input, sizeof(float)*NUM_THREADS*2));

}
