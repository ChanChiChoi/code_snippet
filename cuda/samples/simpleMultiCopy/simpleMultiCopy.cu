#include<helper_cuda.h>

/*
 * Quadro and Tesla GPUs 中计算能力 >= 2.0 的 can overlap two memcopies
 * with kernel execution. This sample illustrates the usage of CUDA streams to
 * achieve overlapping of kernel execution with copying data to and from the device.
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5
 * microsecond precision.
 *
 * Elapsed times are averaged over nreps repetitions (10 by default).
 *
*/
#include<iostream>
#include<helper_cuda.h>
using namespace std;

#define STREAM_COUNT 4

__global__ void
incKernel(int *g_out, int *g_in, int N, int inner_reps){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<N){
    for(int i=0; i<inner_reps; i++)
      g_out[idx] = g_in[idx] + 1;
  }
}


void 
init(int *h_data_source, int *h_data_in[],int N, int memsize){
  for(int i=0;i<N; i++)
    h_data_source[i]=0;
  for(int i=0; i<STREAM_COUNT; i++)
    memcpy(h_data_in[i], h_data_source, memsize);
}

int
main(int argc, char *argv[]){

  int devID = 0;
  float scale;
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, devID));

  cout<<prop.name<<" has "<<prop.multiProcessorCount
      <<" MP(s) x "<<_ConvertSMVer2Cores(prop.major, prop.minor)
      <<" (Cores/MP) = "<<_ConvertSMVer2Cores(prop.major, prop.minor)*prop.multiProcessorCount
      <<" (Cores)"<<endl;
  cout<<"当前GPU是否支持在GPU执行kernel时候，同时执行一个CPU与GPU之间数据传输: "
      <<(prop.deviceOverlap?"Yes":"NO")<<endl;
  cout<<"当前GPU是否支持在GPU执行kernel时候，同时执行两个CPU与GPU之间数据传输: "
      <<(prop.major>=2 && prop.asyncEngineCount>1?"Yes":"NO")<<endl;

  //小于32个核的需要缩小负载
  auto data1 = _ConvertSMVer2Cores(prop.major, prop.minor)*\
              (float)prop.multiProcessorCount;
  scale = max(32.0f/data1, 1.0f);
  cout<<"scale:"<<scale<<endl;
  int N = 1<<22;
  N = (int)((float)N/scale);

  cout<<"array_size:"<<N<<endl;

  int memsize = N*sizeof(int);
  int inner_reps = 5;
  dim3 block{512};
  int thread_blocks=N/block.x;
  dim3 grid(thread_blocks%65535, thread_blocks/65535+1);
  
  //分配资源
  int *h_data_source = (int*)malloc(memsize);
  int *h_data_sink = (int*)malloc(memsize);
  int *h_data_in[STREAM_COUNT];
  int *d_data_in[STREAM_COUNT];
  int *h_data_out[STREAM_COUNT];
  int *d_data_out[STREAM_COUNT];

  cudaStream_t stream[STREAM_COUNT];
  cudaEvent_t cycleDone[STREAM_COUNT];
  cudaEvent_t st, ed;

  for(int i=0; i<STREAM_COUNT; i++){

    checkCudaErrors(cudaHostAlloc(&h_data_in[i],memsize,cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_data_in[i], memsize));

    checkCudaErrors(cudaHostAlloc(&h_data_out[i],memsize,cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_data_out[i], memsize));

    checkCudaErrors(cudaStreamCreate(&stream[i]));   
    checkCudaErrors(cudaEventCreate(&cycleDone[i]));

    cudaEventRecord(cycleDone[i], stream[i]);
  }

  cudaEventCreate(&st);
  cudaEventCreate(&ed);

  init(h_data_source, h_data_in,N, memsize);

  //暖核
  incKernel<<<grid, block>>>(d_data_out[0], d_data_in[0], N, inner_reps);
  // --------- h2d
  cudaEventRecord(st,0);
  checkCudaErrors(cudaMemcpyAsync(d_data_in[0],h_data_in[0], memsize,
                             cudaMemcpyHostToDevice, 0));
  cudaEventRecord(ed,0);
  cudaEventSynchronize(ed);
  float memcpy_h2d_time;
  cudaEventElapsedTime(&memcpy_h2d_time,st,ed);
  cout<<"memcpy_h2d_time:"<<memcpy_h2d_time<<" ms; "
       <<(memsize*1e-6)/memcpy_h2d_time<<" GB/s"<<endl;

  //---d2h
  cudaEventRecord(st,0);
  checkCudaErrors(cudaMemcpyAsync(h_data_out[0],d_data_out[0],memsize,
                                cudaMemcpyDeviceToHost,0));
  cudaEventRecord(ed,0);
  cudaEventSynchronize(ed);
  float memcpy_d2h_time;
  cudaEventElapsedTime(&memcpy_d2h_time,st,ed);
  cout<<"memcpy_d2h_time:"<<memcpy_d2h_time<<" ms; "
      <<(memsize*1e-6)/memcpy_d2h_time<<" GB/s"<<endl;

  //----kernel
  cudaEventRecord(st,0);
  incKernel<<<grid, block>>>(d_data_out[0], d_data_in[0], N, inner_reps);
  cudaEventRecord(ed,0);
  cudaEventSynchronize(ed);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time,st,ed);
  cout<<"kernel run:"<<kernel_time<<" ms; "
      <<(inner_reps*memsize*2e-6)/kernel_time<<" GB/s"<<endl;
  //--------
  
  
  
}
