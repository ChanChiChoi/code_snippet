#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<iostream>
#include<string>

#include<cuda.h>
#include<helper_cuda_drvapi.h>
#include<drvapi_error_string.h>

using std::cout;
using std::endl;

void check(CUresult &error_id, string &error_str){

  if(error_id != CUDA_SUCCESS){
    cout<<errro_str<<" [code]:"<<error_id
        <<" -> [string]:"<<getCUdaDrvErrorString(error_id);
    cout<<"Result = FAIL"<<endl;
    exit(EXIT_FAILURE);
  }
}


int
main(int argc, char **argv){

  CUdevice dev;
  int major=0, minor=0;
  int deviceCount=0;
  string deviceName;

  cout<<argv[0]<<" Starting..."<<endl;
  // need to link with cuda.lib files on windows os
  cout<<" CUDA Device Query (Driver API) s statically linked version"<<endl;

  CUresult error_id = cuInit{0};
  check(error_id,"cuInit(0) returned ");

  error_id = cuDeviceGetCount(&deviceCount);
  check(error_id,"cuDeviceGetCount returned");
  

  
}

