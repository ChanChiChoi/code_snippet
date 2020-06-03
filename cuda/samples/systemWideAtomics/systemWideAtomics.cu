// 在cpu测的原子算子，诸如下面列出的，都是来自gcc文档
//__sync_add_and_fetch
//__sync_and_and_fetch
//__sync_bool_compare_and_swap
//__sync_fetch_and_add
//__sync_fetch_and_and
//__sync_fetch_and_nand
//__sync_fetch_and_or
//__sync_fetch_and_sub
//__sync_fetch_and_xor
//__sync_lock_release
//__sync_lock_test_and_set
//__sync_nand_and_fetch
//__sync_or_and_fetch
//__sync_sub_and_fetch
//__sync_synchronize
//__sync_val_compare_and_swap
//__sync_xor_and_fetch
//__thread
//************************************************8
#include<math.h>
#include<cstdio>
#include<ctime>
#include<stdint.h>
#include<iostream>
#include<cuda_runtime.h>
#include<helper_cuda.h>

#define LOOP_NUM 50

using std::cout;
using std::endl;
using std::cerr;


__global__ void
atomicKernel(int *atom_arr){
  unsigned int tid = blockDim.x*blockIdx.x+threadIdx.x;
  for(int i=0; i<LOOP_NUM; i++){

    atomicAdd_system(&atom_arr[0],10);
    atomicExch_system(&atom_arr[1],tid);
    atomicMax_system(&atom_arr[2],tid);
    atomicMin_system(&atom_arr[3],tid);

    // eads the 32-bit word old located at the address address in global or shared memory,
    // computes ((old >= val) ? 0 : (old+1)), and stores the result back to memory at 
    // the same address. These three operations are performed in one atomic transaction. 
    // The function returns old.
    atomicInc_system((unsigned int *)&atom_arr[4],17);
    // (((old == 0) || (old > val)) ? val : (old-1) )
    atomicDec_system((unsigned int *)&atom_arr[5],137);
    //  (old == compare ? val : old)
    atomicCAS_system(&atom_arr[6],tid-1,tid);

    atomicAnd_system(&atom_arr[7],2*tid+7);
    atomicOr_system(&atom_arr[8],1<<tid);
    atomicXor_system(&atom_arr[9],tid);
  }

}

void
atomicKernel_CPU(int *atom_arr, int nThreads){
  for(int i=nThreads; i<2*nThreads; i++){
    for(int j=0; j<LOOP_NUM; j++){
      __sync_fetch_and_add(&atom_arr[0],10); //add
      __sync_lock_test_and_set(&atom_arr[1],i); // exchange

      //max
      int old, expected;
      do{
         expected = atom_arr[2];
         old = __sync_val_compare_and_swap(&atom_arr[2],expected, max(expected, i));
      }while(old != expected);

      //min
      do{
        expected = atom_arr[3];
        old = __sync_val_compare_and_swap(&atom_arr[3],expected, min(expected, i));
      }while(old!= expected);

      //increment (modulo 17+1)
      int limit = 17;
      do{
        expected = atom_arr[4];
        old = __sync_val_compare_and_swap(&atom_arr[4],expected,
                        (expected>=limit)?0:expected+1);
      }while(old != expected);

      //decrement
      limit = 137;
      do{
        expected = atom_arr[5];
        old = __sync_val_compare_and_swap(&atom_arr[5],expected,
                              ((expected==0)||(expected>limit))?limit:expected-1);
      }while(old != expected);

      // compare and swap
      __sync_val_compare_and_swap(&atom_arr[6], i-1, i);

      // and
      __sync_fetch_and_and(&atom_arr[7], 2*i+7);
      // or
      __sync_fetch_and_or(&atom_arr[8], 1<<i);
      // xor
      // 11th 元素应该是0xff
      __sync_fetch_and_xor(&atom_arr[9],i);
    }   
  }
}

int
main(int argc, char*argv[]){
  cudaDeviceProp prop;
  int idev = 0;
  checkCudaErrors(cudaGetDeviceProperties(&prop, idev));

  if(!prop.managedMemory){
    cerr<<"Unified memory not supported on this device"<<endl;
    exit(EXIT_FAILURE);   
  }
  if(prop.computeMode == cudaComputeModeProhibited){
    cerr<<"this sample requires a device in either default or process exclusive mode"<<endl;
    exit(EXIT_FAILURE);   
  }
  if(prop.major < 6){
    cerr<<"this sample requires a minimum CUDA compute 6.0 capablity"<<endl;
    exit(EXIT_FAILURE);   
  }

  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 10;
  int *atom_arr;
  if(prop.pageableMemoryAccess){
    cout<<"CAN access pageable memory"<<endl;
    atom_arr = (int *)malloc(sizeof(int)*numData);
  }else{
    cout<<"CANNOT access pageable memory"<<endl;
    checkCudaErrors(cudaMallocManaged(&atom_arr, sizeof(int)*numData));
  }
  //--------------------------
  for(unsigned int i=0; i<numData; i++)
    atom_arr[i] = 0;
  //为了让AND 和XOR测试能够生成不是0的结果
  atom_arr[7] = atom_arr[9] = 0xff;

  atomicKernel<<<numBlocks, numThreads>>>(atom_arr);
  checkCudaErrors(cudaDeviceSynchronize());

  for(unsigned int i=0; i<numData; i++)
    cout<<i<<": "<< atom_arr[i]<<endl;
  //---------------------------
  cout<<"=============================="<<endl;
  for(unsigned int i=0; i<numData; i++)
    atom_arr[i] = 0;
  //为了让AND 和XOR测试能够生成不是0的结果
  atom_arr[7] = atom_arr[9] = 0xff;

  atomicKernel_CPU(atom_arr, numBlocks*numThreads);

  for(unsigned int i=0; i<numData; i++)
    cout<<i<<": "<< atom_arr[i]<<endl;
  //---------------------------
  if(prop.pageableMemoryAccess){
    free(atom_arr);
  }else{
    cudaFree(atom_arr);
  }

  cout<<"systemWideAtomic completed"<<endl;
}
