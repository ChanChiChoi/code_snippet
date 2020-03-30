#include<iostream>
#include<stdio.h>

#include<helper_cuda.h>

using namespace std;
//所有kernel都能访问到
__device__ int g_uids = 0;

__device__ void
print_info(int depth, int thread, int uid, int parent_uid){
  // 第0号线程 发起DP
  if(threadIdx.x == 0){
    if(depth == 0)
      printf("BLOCK %d launched by the host\n",uid);
    else{
      char buffer[32];
      for(int i=0; i<depth; i++){
         buffer[3*i+0] = '|';
         buffer[3*i+1] = ' ';
         buffer[3*i+2] = ' ';
      }
      buffer[3*depth] = '\0';
      printf("%s Depth[%d] BLOCK %d launched by block,thread:[%d,%d]\n", 
            buffer, depth,uid, parent_uid,thread);
    }
  }
  __syncthreads();
}


__global__ void
cdp_kernel(int max_depth, int depth, int thread, int parent_uid){

  //为了让block中thread共享，但是只有tid=0的才能set; s_uid 表示块id
  __shared__ int s_uid;
  if(threadIdx.x == 0){
    s_uid = atomicAdd(&g_uids, 1);
  }
  __syncthreads();

  print_info(depth, thread,s_uid,parent_uid);

  if(++depth >= max_depth)
   return;

  cdp_kernel<<<gridDim,blockDim>>>(max_depth, depth, threadIdx.x, s_uid);
}


int main(int argc, char const *argv[]){

  cout<<"starting CUDA Dynamic Parallelism"<<endl;
  int max_depth = 2;
  
  if(checkCmdLineFlag(argc, argv,"help") || 
     checkCmdLineFlag(argc, argv,"h")){
     cout<<"Usage: "<<argv[0]
         <<" depth=<max_depth>\t(where max_depth is a value between 1 and 8)."<<endl;
     exit(EXIT_SUCCESS);
  }
  
  if(checkCmdLineFlag(argc, argv, "depth")){
    max_depth = getCmdLineArgumentInt(argc, argv, "depth");
    if (max_depth<1 || max_depth>8){
      cout<<"depth parameter has to be between 1 and 8"<<endl;
      exit(EXIT_FAILURE);
    }
  }

  //================================
  // find/set the device
  int device_count = 0, device = -1;
  checkCudaErrors(cudaGetDeviceCount(&device_count));
  for(int i=0; i<device_count; i++){
    cudaDeviceProp properties;
    checkCudaErrors(cudaGetDeviceProperties(&properties, i));

    if(properties.major>3 || (properties.major == 3 && properties.minor >= 5)){
      device = i;
      cout<<"Running on GPU:"<<device<<" ("<<properties.name<<") "<<endl;
      break;
    }else{
      cout<<"ERROR: dynamic parallelism requires GPU devices with compute SM 3.5 or higher"<<endl;
      cout<<"Current GPU device["<<i<<"] has compute SM"<<properties.major<<"."<<properties.minor
          <<endl<<"Exiting..."<<endl;
      continue;
    }
  } 
 
  if(device == -1){
    cerr<<"dynamic parallelism requires GPU devices with compute SM 3.5 or higher"<<endl
        <<"Exiting..."<<endl;
    exit(EXIT_FAILURE);
  }
  //设置第一个符合的设备
  cudaSetDevice(device);

  cout<<"====================================="<<endl;
  cout<<"The CPU launches kernel configuration<<<2,2>>> "<<endl
      <<"on the deviec each thread will launch kernel<<<2,2>>>"<<endl
      <<"The GPU we will do that recursively, until reaches"<<endl
      <<"max_depth="<<max_depth<<endl<<endl;
  

  cout<<"2";
  int num_blocks = 2, sum = 2;
  for(int i=1; i<max_depth; i++){
    num_blocks *= 4;
    cout<<"+"<<num_blocks;
    sum += num_blocks;
  }
  cout<<"="<<sum<<" blocks are launched ("<<sum-2<<" from GPU)"<<endl;
  cout<<"===============================ready run"<<endl;
 
  //限制CDP递归的深度
  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

  //host侧运行kernel
  cout<<"Launching cdp_kernel with CUDA Dynamic Parallelism"<<endl;
  cdp_kernel<<<2,2>>>(max_depth, 0, 0, -1);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  exit(EXIT_SUCCESS);
  
}
