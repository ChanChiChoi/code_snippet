#include<cstdio>
#include<iostream>
#include<cassert>
#include<helper_cuda.h>


#include<chrono>
#include"simpleMultiGPU.h"
using std::cout;
using std::endl;
using namespace std::chrono;

constexpr int MAX_GPU_COUNT = 32;
constexpr int DATA_N = 1048576*32;

__global__ static void
reduceKernel(float *d_result, float *d_input, int N){
  //一个数值矩阵，第一行的个数为线程个数，然后第一行每个线程各自计算自己
  //这一列的数据
  const int tid = blockIdx.x*blockDim.x+threadIdx.x; // 一行
  const int threadN = gridDim.x*blockDim.x;// 一行线程
  float sum = 0;
  for(int i = tid; i<N; i+= threadN)// 这行线程每个线程往下计算
   sum += d_input[i];

  d_result[tid] = sum;
}

int main(int argc, char*argv[]){

  TGPUplan plan[MAX_GPU_COUNT]; //最大支持32个gpu
  float h_SumGPU[MAX_GPU_COUNT]; //最大支持32个gpu

//  float sumGPU;
//  double sumCPU, diff;
  constexpr int block = 32;
  constexpr int thread = 256;
  constexpr int ACCUM_N = block*thread;
  
  int gpu_num;
  checkCudaErrors(cudaGetDeviceCount(&gpu_num));
  if(gpu_num > MAX_GPU_COUNT)
    gpu_num = MAX_GPU_COUNT;
  cout<<"可用的设备有:"<<gpu_num<<" 个"<<endl;

  cout<<"生成数据"<<endl;
  // 平均分配空间
  for(int i=0; i<gpu_num; i++)
    plan[i].dataN = DATA_N/gpu_num;
  //把余数分完
  for(int i=0; i<DATA_N%gpu_num; i++)
    plan[i].dataN++;
  
  int gpuBase = 0;
  for(int i=0; i<gpu_num; i++){
    plan[i].h_Sum = h_SumGPU + i; //分配在h_SumGPU 数组的位置
    gpuBase += plan[i].dataN;
  }

  for(int i=0; i<gpu_num; i++){
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaStreamCreate(&plan[i].stream));
    //分配内存
    checkCudaErrors(cudaMalloc((void**)&plan[i].d_Data,
                                plan[i].dataN*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&plan[i].d_Sum,
                               ACCUM_N*sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&plan[i].h_Sum_from_device,
                                  ACCUM_N*sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&plan[i].h_Data, 
                                   plan[i].dataN*sizeof(float)));
    //随机初始化
    for(int j=0; j<plan[i].dataN; j++)
      plan[i].h_Data[j] = (float)rand()/(float)RAND_MAX;

  }
  
  auto st = system_clock::now();

  //将数据从cpu拷贝到gpu，
  for(int i=0; i<gpu_num; i++){
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaMemcpyAsync(plan[i].d_Data,
                                    plan[i].h_Data,
                                    plan[i].dataN*sizeof(float),
                                    cudaMemcpyHostToDevice, 
                                    plan[i].stream));
     reduceKernel<<<block, thread,0,plan[i].stream>>>(plan[i].d_Sum,
                                                      plan[i].d_Data,
                                                      plan[i].dataN);
     getLastCudaError("reduceKernel() executaion failed\n");
     checkCudaErrors(cudaMemcpyAsync(plan[i].h_Sum_from_device,
                                     plan[i].d_Sum,
                                     ACCUM_N*sizeof(float),
                                     cudaMemcpyDeviceToHost,
                                     plan[i].stream));

  }

  for(int i=0; i<gpu_num; i++){
    float sum=0;
    checkCudaErrors(cudaSetDevice(i));
    cudaStreamSynchronize(plan[i].stream);
    for(int j=0; j<ACCUM_N; j++)
      sum+= plan[i].h_Sum_from_device[j];
    *(plan[i].h_Sum) = (float)sum; // 将结果写入到h_SumGPU对应位置

    checkCudaErrors(cudaFreeHost(plan[i].h_Sum_from_device));
    checkCudaErrors(cudaFree(plan[i].d_Sum));
    checkCudaErrors(cudaFree(plan[i].d_Data));
    checkCudaErrors(cudaStreamDestroy(plan[i].stream));
  }

  float sumGPU = 0;
  for(int i=0; i<gpu_num; i++)
    sumGPU += h_SumGPU[i];
  auto ed = system_clock::now();
  //cout<<" GPU 处理时间:"<<difftime(ed,st)<<" s"<<endl;
  cout<<" GPU 处理时间:"<<duration_cast<milliseconds>(ed-st).count()<<" ms"<<endl;

  for(int i=0;i<gpu_num;i++){
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaFreeHost(plan[i].h_Data));
  }
} 
