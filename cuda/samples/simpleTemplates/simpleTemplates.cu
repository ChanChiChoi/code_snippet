#include<cstdio>
#include<cassert>
#include<cstring>
#include<cmath>

#include<helper_cuda.h>
#include<timer.h>

#include "sharedmem.cuh"


template<typename T> void
runTest(int argc, char *argv[], int len, int g_TotalFailres){
  
  unsigned int num_threads = len;
  unsigned int mem_size = sizeof(float)*num_threads;

  // 分配host内存并初始化
  T *h_idata = (T*)malloc(mem_size);
  for(unsigned int i=0; i<num_threads; i++){
    h_idata[i] = (T)i;
  }  

  //分配device的输入内存并复制
  T *d_idata;
  checkCudaErrors(cudaMalloc((void**)&d_idata, mem_size));
  checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));
  //分配device的输出内存
  T *d_odata;
  checkCudaErrors(cudaMalloc((void**)&d_odata, mem_size));
  // kernel调度
  dim3 grid(1,1,1);
  dim3 threads(num_threads,1,1);

  testKernel<T><<<grid,threads,mem_size>>>(d_idata, d_odata);

  getLastCudaError("Kernel execution failed");
  // 分配host侧的结果内存,然后结果cp到host
  T *h_odata = (T*)malloc(mem_size); 
  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(T)*num_threads,
                             cudaMemcpyDeviceToHost));
  
  
}


int
main(int argc, char *argv[]){

  int g_TotalFailres=0;
  cout<<"> runTest<float, 32>"<<endl;
  runTest<float>(argc, argv, 32,g_TotalFailres);
  cout<<"> runTest<int, 64>"<<endl;
  runTest<int>(argc, argv, 64, g_TotalFailres);

  cout<<"[simpleTemplates] -> Test Results: "<<g_TotalFailres<<" Failures"<<endl;
  exit(g_TotalFailres == 0? EXIT_SUCCESS: EXIT_FAILURE);
}

//需要使用模板特例化来解决针对不同类型下CUTIL的数组对比和文件写功能
