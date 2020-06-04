// __any_sync: 是在这个warp中，如果有一个计算的结果为非0，
// 则这个warp中所有该行返回的都是1
// 如下面这个例子，d_o所属的这warp个结果都为1或者0
//***********************************
// #include<iostream>
// using namespace std;
// 
// __global__ void
// test(int *d_i, int *d_o){
//   int tid = threadIdx.x;
//   d_o[tid] = __all_sync(0xffffffff, d_i[tid]);
//   
// }
// 
// int main(){
// 
//  int *d_i, *d_o;
//  dim3 grid(1);
//  dim3 blocks(32);
// 
//  int *h_i = (int*)malloc(grid.x*blocks.x*sizeof(int));
//  for(int i=0;i<blocks.x;i++)
//    h_i[i] = 0;
// 
//  h_i[0]=9;
// 
//  cudaMalloc((void**)&d_i, grid.x*blocks.x*sizeof(int));
//  cudaMalloc((void**)&d_o, grid.x*blocks.x*sizeof(int));
//  cudaMemcpy(d_i, h_i, sizeof(int)*grid.x*blocks.x, cudaMemcpyHostToDevice);
//  test<<<grid,blocks>>>(d_i,d_o);
//  
//   int *h_o = (int*)malloc(grid.x*blocks.x*sizeof(int));
//   cudaMemcpy(h_o,d_o,sizeof(int)*grid.x*blocks.x, cudaMemcpyDeviceToHost);
//   for(int i=0; i<blocks.x; i++)
//    cout<<h_o[i]<<endl;
// }
//*****************************
#include<cstdio>
#include<cassert>
#include<iostream>
#include<helper_cuda.h>

#include "vote_kernel.cuh"

#define VOTE_DATA_GROUP 4

using std::cout;
using std::endl;


void
genVoteTestPattern(unsigned int *VOTE_BATTERN, int size){
 //  一共有4个warp,所以这里创建了4个完全不同的warp
  //第一个 for testing VOTE.any (所有线程都会返回0)
  for(int i=0; i<size/4; i++) 
    VOTE_BATTERN[i] = 0x00000000;

  // for tesring VOTE.any (1/2的线程返回1)
  for(int i=size/4; i<2*size/4; i++)
    VOTE_BATTERN[i] = (i&0x01) ? i : 0; // 奇偶不同

  //for testing VOTE.all
  for(int i=2*size/4; i<3*size/4; i++)
    VOTE_BATTERN[i] = (i&0x01)? 0: i; //奇偶不同

  //for testing VOTE.all
  for(int i=3*size/4; i<4*size/4; i++)
    VOTE_BATTERN[i] = 0xffffffff; 
}

int
main(int argc, char *argv[]){
  unsigned int *h_input, *h_result;
  unsigned int *d_input, *d_result;
  
  bool *dinfo = nullptr, *hinfo = nullptr;
  //int error_count[3] = {0, 0, 0};
  cudaDeviceProp prop;
  int devID=0, warp_size = 32;

  cout<<argv[0]<<endl;

  checkCudaErrors(cudaGetDeviceProperties(&prop, devID));
  cout<<"> GPU device has"<<prop.multiProcessorCount<<" Multi-Processors, SM "
      <<prop.major<<"."<<prop.minor<<" compute capabilities"<<endl;
  //------------------------ 
  h_input = (unsigned int *)malloc(VOTE_DATA_GROUP*warp_size*sizeof(unsigned int));
  h_result = (unsigned int *)malloc(VOTE_DATA_GROUP*warp_size*sizeof(unsigned int));
  checkCudaErrors(cudaMalloc((void**)&d_input,VOTE_DATA_GROUP*warp_size*sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_result, VOTE_DATA_GROUP*warp_size*sizeof(int)));
  
  genVoteTestPattern(h_input, VOTE_DATA_GROUP*warp_size);
  checkCudaErrors(cudaMemcpy(d_input, h_input, VOTE_DATA_GROUP*warp_size*sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

  //----------------------------
  cout<<"==========================="<<endl;
  cout<<"VOTE Kernel Test 1/3..."<<endl;
  cout<<"Running <<VOTE.any>> kernel1..."<<endl;
  {
    checkCudaErrors(cudaDeviceSynchronize());
    dim3 grid(1,1);
    dim3 blocks(VOTE_DATA_GROUP*warp_size,1);
    VoteAnyKernel1<<<grid, blocks>>>(d_input, d_result, VOTE_DATA_GROUP*warp_size);
    getLastCudaError("VoteAnyKernel() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
  }
  checkCudaErrors(cudaMemcpy(h_result, d_result, VOTE_DATA_GROUP*warp_size*sizeof(unsigned int),
                             cudaMemcpyDeviceToHost)); 
  cout<<"期望结果：第一行皆为0，后面几行皆为1"<<endl;
  for(int i=0; i<VOTE_DATA_GROUP; i++){
     for(int j=0; j<warp_size; j++){
       cout<<h_result[i*warp_size+j]<<" ";
     }
     cout<<endl;
  }

  //----------------------------
  cout<<"==========================="<<endl;
  cout<<"VOTE Kernel Test 2/3..."<<endl;
  cout<<"Running <<VOTE.all>> kernel2..."<<endl;
  {
    checkCudaErrors(cudaDeviceSynchronize());
    dim3 grid(1,1);
    dim3 blocks(VOTE_DATA_GROUP*warp_size,1);
    VoteAllKernel2<<<grid, blocks>>>(d_input, d_result, VOTE_DATA_GROUP*warp_size);
    getLastCudaError("VoteAllKernel() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
  }
  checkCudaErrors(cudaMemcpy(h_result, d_result, VOTE_DATA_GROUP*warp_size*sizeof(unsigned int),
                             cudaMemcpyDeviceToHost)); 
  cout<<"期望结果：前三行皆为0，后面一行皆为1"<<endl;
  for(int i=0; i<VOTE_DATA_GROUP; i++){
     for(int j=0; j<warp_size; j++){
       cout<<h_result[i*warp_size+j]<<" ";
     }
     cout<<endl;
  }

  //-----------------------------
  hinfo = (bool *)calloc(warp_size*3*3,sizeof(bool));
  checkCudaErrors(cudaMalloc((void**)&dinfo, warp_size*3*3*sizeof(bool)));
  checkCudaErrors(cudaMemcpy(dinfo, hinfo ,warp_size*3*3*sizeof(bool),cudaMemcpyHostToDevice));

  cout<<"==========================="<<endl;
  cout<<"VOTE Kernel Test 3/3..."<<endl;
  cout<<"Running <<VOTE.any>> kernel3..."<<endl;
  {
    checkCudaErrors(cudaDeviceSynchronize());
    VoteAnyKernel3<<<1, warp_size*3>>>(dinfo, warp_size);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  checkCudaErrors(cudaMemcpy(hinfo, dinfo, warp_size*3*3*sizeof(bool), cudaMemcpyDeviceToHost));

  cout<<"期望结果：前三行皆为0"<<endl;
  for(int i=0; i<warp_size*3; i++){
     cout<<"["<<i<<"]:";
     for(int j=0; j<3; j++){
       cout<<hinfo[i*3+j]<<" ";
     }
     cout<<endl;
  }

  //free
  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_result));
  free(h_input);
  free(h_result);
  free(hinfo);
  checkCudaErrors(cudaFree(dinfo));
  
  cout<<"Shhutting down..."<<endl;
  





 
}
