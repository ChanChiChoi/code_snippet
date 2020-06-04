#include<iostream>
#include<cooperative_groups.h>

#include<helper_cuda.h>

using namespace cooperative_groups;


__device__ int 
sumReduction(thread_group g, int *x, int val){
  //获取当前线程的索引
  int lane = g.thread_rank();

/*
  for(int i = g.size()/2; i>0 ;i /= 2){
    x[lane] = val;
    g.sync();

    if(lane<i){
      val += x[lane +i];
    }
    g.sync();
  }
*/
  //先进行赋值
  x[lane] = val;
  g.sync();

  for(int i = g.size()/2; i>0 ;i /= 2){

    if(lane<i){
      x[lane] += x[lane +i];
    }
    g.sync();
  }

  if(lane == 0)
    return x[lane];
  else
    return -1;
}

__global__ void
cgKernel(){
  //获取当前块内所有线程
  thread_block threadBlockGroup = this_thread_block();
  int threadBlockGroupSize = threadBlockGroup.size();

  extern __shared__ int workspace[];

  int input, output, expectedOutput;

  //获取当前线程在块中的索引
  input = threadBlockGroup.thread_rank();
  int cgTid = input;
  
  //首项加末项乘以项数(0 + n-1)*(n)/2 :(0+63)*64/2=2016
  expectedOutput = (threadBlockGroupSize-1)*threadBlockGroupSize/2;
  
  output = sumReduction(threadBlockGroup, workspace, input);

  if(cgTid == 0){
    printf("index inside this group from 0 to %d, result is %d, expected is %d\n",
             threadBlockGroupSize-1, output, expectedOutput);
    printf(" need created %d groups, each group has 16 threads\n",threadBlockGroupSize/16);
  }

  threadBlockGroup.sync();
  //将当前group以每个16个线程进行划分
  thread_block_tile<16> tiledPartition16 = tiled_partition<16>(threadBlockGroup);

  int  cgSubTid =  tiledPartition16.thread_rank();
  //整个group中当前线程索引，减去分块之后线程内部索引，
  int workspaceOffset = cgTid - cgSubTid;

  input = cgSubTid;
  expectedOutput = (0+15)*16/2; // 首项加末项 乘以项数 除以2
  output = sumReduction(tiledPartition16, workspace+workspaceOffset, input);

  if(cgSubTid == 0){
    printf("index inside this titledPartition16 group from 0 to 15, result is %d, expected is %d\n",output, expectedOutput);
  }
  
}


int
main(){
  
  int blocksPerGrid = 1;
  int threadsPerBlock = 64;
  cgKernel<<<blocksPerGrid, threadsPerBlock,threadsPerBlock*sizeof(int)>>>();
  checkCudaErrors(cudaDeviceSynchronize());
  return 0;
}
