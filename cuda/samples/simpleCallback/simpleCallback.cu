/*
 * This sample implements multi-threaded heterogeneous computing workloads with the new CPU callbacks for CUDA streams and events introduced with CUDA 5.0.
 * Together with the thread safety of the CUDA API implementing heterogeneous workloads that float between CPU threads and GPUs has become simple and efficient.
 *
 * The workloads in the sample follow the form CPU preprocess -> GPU process -> CPU postprocess.
 * Each CPU processing step is handled by its own dedicated thread. GPU workloads are sent to all available GPUs in the system.
 *
 */

#include<iostream>
#include<memory>
#include<helper_cuda.h>

#include "multithreading.h"
using namespace std;

int const N_workloads = 8;
int const N_ele_per_workload = 100000;


struct heterogeneous_workload{
  int id;
  int cudaDveiceID;
  int *h_data;
  int *d_data;
  cudaStream_t stream;
  bool success;
};


void CUDART_CB
myStreamCallback(cudaStream_t stream, cudaError_t status, void *data);





void CUDART_CB
myStreamCallback(cudaStream_t stream, cudaError_t status, void *data){
  checkCudaErrors(status);//检查stream操作之后的状态
  cutStartThread(postprocess,data);//spawn新的cpu worker 继续做后处理
}


// simpleCallback only supports max 32 GPU(s)
int
main(int argc, char *argv[]){
  int N_gpus, max_gpus = 0;
  int gpuInfo[32];

  checkCudaErrors(cudaGetDeviceCount(&N_gpus));// 为了确定N_gpus
  if(N_gpus>32)
    cerr<<"simpleCallback only supports max 32 GPU(s)"<<endl;
  //为了确定当前机器最大可用gpu数量
  for(int i=0; i<N_gpus; i++){
    int sm;
    cudaDeviceProp prop;
    cudaSetDevice(i);
    cudaGetDeviceProperties(&prop, i);
    sm = prop.major<<4+prop.minor;
    if(sm>=0x11){
      cout<<"card:"<<i<<" support callback functions"<<endl;
      gpuInfo[max_gpus++] = i;
    }
  }


  unique_ptr<heterogeneous_workload[]>workloads{
                                  new heterogeneous_workload[N_workloads]};
  thread_barrier = cutCreateBarrier(N_workloads);  

  //主线程为每个异质workload 各分出一个CPU worker线程
  cout<<"starting "<<N_workloads<<" heterogeneous computing workloads"<<endl;
  
  for(int i=0; i<N_workloads; i++){
    workloads[i].id = i;
    workloads[i].cudaDveiceID = gpuInfo[i%max_gpus];
    cutStartThread(launch, &workloads[i].get());
  }
  cuWaitForBarrier(&thread_barrier);
  cout<<"total of "<<N_workloads<<" workloads finisned"<<endl;

  bool success = true;
  for(int i=0; i<N_workloads; i++)
    success &= workloads[i].success;
  
  cout<<(success ? "Success" : "Failure")<<endl;
  exit(success ? EXIT_SUCCESS : EXIT_FAILURE);
}
