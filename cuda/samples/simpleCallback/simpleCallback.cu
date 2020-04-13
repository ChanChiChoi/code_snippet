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

CUTBarrier thread_barrier;

struct heterogeneous_workload{
  int id;
  int cudaDeviceID;
  int *h_data;
  int *d_data;
  cudaStream_t stream;
  bool success;
};


void CUDART_CB
myStreamCallback(cudaStream_t stream, cudaError_t status, void *data);

__global__ void
incKernel(int *data, int N){
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid<N)
    data[tid]++;
}

CUT_THREADPROC launch(void *void_arg){
  heterogeneous_workload *workload = (heterogeneous_workload*) void_arg;
  //为当前线程选择GPU
  checkCudaErrors(cudaSetDevice(workload->cudaDeviceID));

  //分配资源
  checkCudaErrors(cudaStreamCreate(&workload->stream));
  checkCudaErrors(cudaMalloc(&workload->d_data, N_ele_per_workload*sizeof(int)));
  checkCudaErrors(cudaHostAlloc(&workload->h_data, 
                                N_ele_per_workload*sizeof(int),
                                cudaHostAllocPortable));

  //cpu线程生成数据
  for(int i=0; i<N_ele_per_workload; i++)
   workload->h_data[i] = workload->id+i;

  //kernel configuration
  dim3 block(512);
  dim3 grid((N_ele_per_workload+block.x-1)/block.x);

  checkCudaErrors(cudaMemcpyAsync(workload->d_data,
                                  workload->h_data,
                                  N_ele_per_workload*sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  workload->stream));
  incKernel<<<grid,block,0,workload->stream>>>(workload->d_data,
                                              N_ele_per_workload);
  checkCudaErrors(cudaMemcpyAsync(workload->h_data,
                                  workload->d_data,
                                  N_ele_per_workload*sizeof(int),
                                  cudaMemcpyDeviceToHost,
                                  workload->stream));

//  checkCudaErrors(cudaLaunchHostFunc(workload->stream, 
//                                     myStreamCallback, 
//                                     workload));
  checkCudaErrors(cudaStreamAddCallback(workload->stream,
                                        myStreamCallback,
                                        workload,
                                        0));
  CUT_THREADEND;
  
}

CUT_THREADPROC postprocess(void *void_arg){
  heterogeneous_workload *workload = (heterogeneous_workload*) void_arg;
  // ...GPU完成了任务，后续工作交给CPU
  checkCudaErrors(cudaSetDevice(workload->cudaDeviceID));
  workload->success = true;
  for(int i=0; i<N_workloads;i++)
    workload->success &= workload->h_data[i] == i+workload->id+1;

  //释放资源
  checkCudaErrors(cudaFree(workload->d_data));
  checkCudaErrors(cudaFreeHost(workload->h_data));
  checkCudaErrors(cudaStreamDestroy(workload->stream));

  //发送信号
  cutIncrementBarrier(&thread_barrier);
  CUT_THREADEND;
}




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
  //创建一个Barrier实例
  thread_barrier = cutCreateBarrier(N_workloads);  //创建屏障同步,将releaseCount设为N

  //主线程为每个异质workload 各分出一个CPU worker线程
  cout<<"starting "<<N_workloads<<" heterogeneous computing workloads"<<endl;
  
  for(int i=0; i<N_workloads; i++){
    workloads[i].id = i;
    workloads[i].cudaDeviceID = gpuInfo[i%max_gpus];
    cutStartThread(launch, (workloads.get()+i));
  }
  //barrier阻塞，保证所有的线程都完成了任务，主线程才执行后续的
  cutWaitForBarrier(&thread_barrier);
  cout<<"total of "<<N_workloads<<" workloads finished"<<endl;

  bool success = true;
  for(int i=0; i<N_workloads; i++)
    success &= workloads[i].success;
  
  cout<<(success ? "Success" : "Failure")<<endl;
  exit(success ? EXIT_SUCCESS : EXIT_FAILURE);
}
