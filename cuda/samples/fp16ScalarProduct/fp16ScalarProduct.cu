#include<iostream>
#include<memory>
#include<cstdlib>
#include<ctime>
#include<vector>

#include<cuda_fp16.h>
#include"helper_cuda.h"

using std::cout;
using std::endl;
using std::unique_ptr;
using std::vector;

using fp = void(*)(int*);

int
main(int argc, char *argv[]){

  srand(time(NULL));
  int const blocks = 128;
  int const threads = 128;
  size_t size = blocks*threads*16; // 一共多少字节

  half2* vec[2];//指针数组
  half2* devVec[2];
  
  vector<unique_ptr<half2,fp>> vec;
  vector<unique_ptr<half2,fp>> devVec;

  float* results;
  float* devResults;

  int devID = 0;
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop,devID));
  if(prop.major<5 || (prop.major==5 && prop.minor<3)){
    cout<<"ERROR: fp16 requires SM 5.3 or higher"<<endl;
    exit(EXIT_FAILURE);
  }

  for(int i=0;i<2;i++){
    half2* tmp;
    checkCudaErrors(cudaMallocHost((void**)&tmp,size*sizeof(half2)))
  }

}

