#include<cmath>
#include<cassert>
#include<cstdio>

#include<cuda_runtime.h>

#include<helper_cuda.h>


#ifndef MAX
#define MAX(a,b) (a>b ?a:b)
#endif

//在cpp文件中定义，并且在cpp中输出的时候也需要c风格形式
extern "C" void
computeGold(char *ref, char *idata, unsigned int const len);

extern "C" void
computeGold2(int2 *ref, int2 *idata, unsigned int char len);

__global__ void 
kernel(int *g_data){

  unsigned int const tid = threadIdx.x;
  //之前是char类型，这里是int类型，也就是一次读取4个char
  int data = g_data[tid];

  //为了避免bank conflicts
  //提取从左到右的8bit，(data<<x)>>24; 
  g_data[tid] = ( (data<<0)>>24 - 10 )<<24;

}


extern "C" bool
runTest(int const argc, char const *argv[], char *data,
        int2 *data_int2, unsigned int len){

  //检测是否刚好是4的倍数
  assert(0 == len(%4));
  unsigned int const num_threads = len / 4;
  unsigned int const mem_size = sizeof(char)*len;
  unsigned int const mem_size_int2 = sizeof(int2)*len;

  //device mem
  char *d_data;
  checkCudaErrors(cudaMalloc((void**)&d_data,mem_size));
  checkCudaErrors(cudaMemcpy(d_data, data, mem_size, cudaMemcpyHostToDevice));

  //for int2 version
  int2 *d_data_int2;
  checkCudaErrors(cudaMalloc((void**)&d_data_int2, mem_size_int2));
  checkCudaErrors(cudaMemcpy(d_data_int2, data_int2, 
                             mem_size_int2,cudaMemcpyHostToDevice));

  dim3 grids(1,1,1);
  dim3 blocks(num_threads,1,1);
  dim3 blocks2(len,1,1);


}
