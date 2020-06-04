#ifndef _SIMPLEATOMICS_KERNEL_H_
#define _SIMPLEATOMICS_KERNEL_H_

__global__ void
kernel(int *g_output){

  unsigned int const tid = blockDim.x*blockIdx.x+threadIdx.x;

  //测试各种原子指令
  
  atomicAdd(&g_output[0], 10);

  atomicSub(&g_output[1],10);

  atomicExch(&g_output[2], tid);//exchange

  atomicMax(&g_output[3], tid);

  atomicMin(&g_output[4], tid);

  atomicInc((unsigned int *)&g_output[5], 17);// modulo 17+1

  atomicDec((unsigned int *)&g_output[6], 137);

  atomicCAS(&g_output[7], tid-1, tid);//compare and swap

  //位处理
  atomicAnd(&g_output[8], 2*tid+7);

  atomicOr(&g_output[9], 1<<tid);

  atomicXor(&g_output[10], tid);

}
#endif
