#ifndef SIMPLEVOTE_KERNEL_CU
#define SIMPLEVOTE_KERNEL_CU

#include<cstdio>
// tests the across-the warp vote(any) intrinsic
// 如果(该warp中)Any个线程返回一个非0值，则该warp中所有线程都会返回一个非0值。
__global__ void
VoteAnyKernel1(unsigned int *input, unsigned int *result, int size){
  int tx = threadIdx.x;
  int mask = 0xffffffff;
  result[tx] = __any_sync(mask, input[tx]);
}

// tests the across-the warp vote(all) intrinsic
// 如果(该warp中)ALL个线程返回一个非0值，则该warp中所有线程都会返回一个非0值。
__global__ void
VoteAllKernel2(unsigned int *input, unsigned int *result, int size){
  int tx = threadIdx.x;
  int mask = 0xffffffff;
  result[tx] = __all_sync(mask, input[tx]);
}

// 一共创建warp_size*3*3个空间，其中前三个warp负责reduce.
__global__ void
VoteAnyKernel3(bool *info, int warp_size){
  int tx = threadIdx.x;
  unsigned int mask = 0xffffffff;
  // 是三个三个一处理的，
  bool *offs = info + (tx*3);

  // offs +0
  // 对于第2,3个warp是为正的，第一个不是
  *offs = __any_sync(mask, (tx>=(warp_size*3)/2));

  // offs +1
  // 对于三个warp的前一半是false，后一半是true
  *(offs+1) = (tx >= (warp_size*3)/2? true: false); 

  // offs +2
  // 只有第三个warp才全为1 
  // if(all((tx >= (warp_size*3)/2)))
  if(__all_sync(0xffffffff,(tx >= (warp_size*3)/2)))
   *(offs+2) = true;
 
  printf("[%d]:%d %d %d\n",tx,*offs, *(offs+1), *(offs+2));
}




#endif
