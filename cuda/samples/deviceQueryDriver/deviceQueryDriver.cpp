#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<iostream>
#include<string>
#include<memory>

#include<cuda.h>
#include<helper_cuda_drvapi.h>
#include<drvapi_error_string.h>

using std::cout;
using std::endl;
using std::unique_ptr;
using std::string;

void check(CUresult &error_id, string const &error_str){

  if(error_id != CUDA_SUCCESS){
    cout<<error_str<<" [code]:"<<error_id
        <<" -> [string]:"<<getCudaDrvErrorString(error_id);
    cout<<"Result = FAIL"<<endl;
    exit(EXIT_FAILURE);
  }
}


int
main(int argc, char **argv){

  CUdevice dev;
  int major=0, minor=0;
  int deviceCount=0;
  unique_ptr<char[]> deviceName{new char[1024]};

  cout<<argv[0]<<" Starting..."<<endl;
  // need to link with cuda.lib files on windows os
  cout<<" CUDA Device Query (Driver API) s statically linked version"<<endl;

  CUresult error_id = cuInit(0);
  check(error_id,"cuInit(0) returned ");

  error_id = cuDeviceGetCount(&deviceCount);
  check(error_id,"cuDeviceGetCount returned ");
  
  if(deviceCount==0){
    cout<<"There are no avaliable device(s) that support CUDA"<<endl;
  }else{
    cout<<"Detected "<<deviceCount<<" CUDA Capable device(s)"<<endl;
  }

  for(dev = 0; dev<deviceCount; dev++){
    error_id = cuDeviceComputeCapability(&major, &minor, dev);
    check(error_id, "cuDeviceComputeCapbility returned ");

    error_id = cuDeviceGetName(deviceName.get(), 256, dev);
    check(error_id, "cuDeviceGetName returned ");

    cout<<"Device: "<<dev<<" "<<deviceName.get()<<endl;

    
  }
  

  
}

