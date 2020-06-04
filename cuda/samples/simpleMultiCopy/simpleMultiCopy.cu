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


void test(cudaEvent_t &st, cudaEvent_t &ed,
          int *d_data_in[], int *h_data_in[],
          int memsize,
          int *d_data_out[], int *h_data_out[],
          dim3 grid, dim3 block,
          int N, int inner_reps
           ){

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
  //--------分析

  cout<<"分析因为传输和重叠带来的加速增益"<<endl;
  cout<<"完全不重叠，即传输，执行，回传:"
      <<memcpy_h2d_time+ kernel_time+ memcpy_d2h_time<<" ms"<<endl;
  cout<<"只允许重叠一个方向:"
      <<max((memcpy_h2d_time+memcpy_d2h_time),kernel_time)<<" ms"<<endl;
  cout<<"两个方向都能重叠:"
      <<max(max(memcpy_h2d_time,memcpy_d2h_time),kernel_time)<<" ms"<<endl;  
  
}

float processWithStream(int nStream, cudaEvent_t &st, cudaEvent_t &ed,
                        int nreps,cudaEvent_t cycleDone[],
                        dim3 grid, dim3 block, cudaStream_t stream[],
                        int*d_data_out[], int *d_data_in[],
                        int N, int inner_reps, int memsize,
                        int*h_data_in[], int *h_data_out[]){

  int current_stream = 0;
  float time;
  cudaEventRecord(st,0);  
  for(int i=0; i<nreps; i++){
    int next_stream = (current_stream+1)%nStream;
    cudaEventSynchronize(cycleDone[next_stream]);

    incKernel<<<grid,block,0,stream[current_stream]>>>(
                                   d_data_out[current_stream],
                                   d_data_in[current_stream],
                                   N,
                                   inner_reps );
    checkCudaErrors(cudaMemcpyAsync(d_data_in[next_stream],
                                    h_data_in[next_stream],
                                    memsize,
                                    cudaMemcpyHostToDevice,
                                    stream[next_stream]));
    checkCudaErrors(cudaMemcpyAsync(h_data_out[current_stream],
                                    d_data_out[current_stream],
                                    memsize,
                                    cudaMemcpyDeviceToHost,
                                    stream[current_stream]));
    checkCudaErrors(cudaEventRecord(cycleDone[current_stream],
                                    stream[current_stream]));
    current_stream = next_stream;
 
  }
  cudaEventRecord(ed,0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&time,st,ed);
  return time;
  
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

  test(st, ed,
       d_data_in, h_data_in,
       memsize,
       d_data_out, h_data_out,
       grid, block,
       N, inner_reps);

  int nreps = 10;
  float serial_time = processWithStream(1, st, ed,
                   nreps,cycleDone,
                   grid, block,stream,
                   d_data_out, d_data_in,
                   N, inner_reps,memsize,
                   h_data_in, h_data_out);
  float overlap_time = processWithStream(STREAM_COUNT, st, ed,
                   nreps,cycleDone,
                   grid, block,stream,
                   d_data_out, d_data_in,
                   N, inner_reps,memsize,
                   h_data_in, h_data_out);
  cout<<"==================="<<endl;
  cout<<"一共运行:"<<nreps<<"次的平均测量时间"<<endl;
  cout<<"完全序列化执行:"<<(serial_time/nreps)<<" ms"<<endl;
  cout<<"使用:"<<STREAM_COUNT<<" 流重叠技术:"<<(overlap_time/nreps)<<" ms"<<endl;
  cout<<"平均加速:"<<(serial_time-overlap_time)/nreps<<" ms"<<endl;
  cout<<"==================="<<endl;
  cout<<"数据吞吐上:"<<endl;
  cout<<"完全序列化:"<<(nreps*(memsize*2e-6)/serial_time)<<" GB/s"<<endl;
  cout<<"使用:"<<STREAM_COUNT<<" 流重叠技术:"<<(nreps*(memsize*2e-6)/overlap_time)<<" GB/s"<<endl;

  free(h_data_source);
  free(h_data_sink);

  for(int i=0; i<STREAM_COUNT;i++){
    cudaFreeHost(h_data_in[i]);
    cudaFree(d_data_in[i]);

    cudaFreeHost(h_data_out[i]);
    cudaFree(d_data_out[i]);
  
    cudaStreamDestroy(stream[i]);
    cudaEventDestroy(cycleDone[i]);
  }
  cudaEventDestroy(st);
  cudaEventDestroy(ed);
 

  


}
