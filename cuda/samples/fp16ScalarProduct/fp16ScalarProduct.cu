#include<iostream>
#include<functional>
#include<memory>
#include<cstdlib>
#include<ctime>
#include<vector>

#include"cuda_fp16.h"
#include"helper_cuda.h"

using std::cout;
using std::endl;
using std::unique_ptr;
using std::vector;
using std::function;

using fp = void(*)(int*);

void genInput(half2 * src, size_t size){
  for(size_t i=0;i <size;i++){
    unsigned int temp = rand();
    temp &= 0x83FF83FF;
    temp |= 0X3C003C00;
    src[i] = *reinterpret_cast<half2*>(&temp);
    //src[i] = *(half2*)&temp;
  }
}


__forceinline__ __device__ void
reduceInShared(half2 * const v){
  int tid = threadIdx.x;
  //两个warp
  if(tid<64)
    v[tid] = __hadd2(v[tid], v[tid+64]);
  __syncthreads();
  //一个warp内部
  if(tid<32)
    v[tid] = __hadd2(v[tid], v[tid+32]);
  __syncthreads();
  if(tid<32)
    v[tid] = __hadd2(v[tid], v[tid+16]);
  __syncthreads();
  if(tid<32)
    v[tid] = __hadd2(v[tid], v[tid+8]);
  __syncthreads();
  if(tid<32)
    v[tid] = __hadd2(v[tid], v[tid+4]);
  __syncthreads();
  if(tid<32)
    v[tid] = __hadd2(v[tid], v[tid+2]);
  __syncthreads();
  if(tid<32)
    v[tid] = __hadd2(v[tid], v[tid+1]);
  __syncthreads();

}

__global__ void
scalarProductKernel(half2 const * const a,
                    half2 const * const b,
                    float * const results,
                    size_t const size
                   ){
  int const stride = gridDim.x*blockDim.x;//一行大小
  __shared__ half2 shArray[128];

  shArray[threadIdx.x] = __float2half2_rn(0.0f);
  half2 value = __float2half2_rn(0.0f);//结果

  //i的取值，blockDim.x 表示块中线程个数；
  //         blockIdx.x 表示块的索引；
  //         threadIdx.x 表示线程的索引
  // 按逻辑应该是thread.x+ blockDim.x*blockIdx.x;
  // 表示经过几个块之后，当前块中线程索引，而不是相加
  // 但是这个例子  核心是展示半精度计算，所以这些逻辑就不纠结了
  for(size_t i = threadIdx.x+blockDim.x+blockIdx.x;
      i<size; i += stride){
    value = __hfma2(a[i], b[i], value);
  }

  shArray[threadIdx.x] = value;// 每个线程同步结果到共享存储
  __syncthreads();

  reduceInShared(shArray);

  //结果写回到results
  if(threadIdx.x == 0){
    half2 res = shArray[0];
    float f_res = __low2float(res) + __high2float(res);
    results[blockIdx.x] = f_res;
  }
}








int
main(int argc, char *argv[]){


  int devID = 0;
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop,devID));
  if(prop.major<5 || (prop.major==5 && prop.minor<3)){
    cout<<"ERROR: fp16 requires SM 5.3 or higher"<<endl;
    exit(EXIT_FAILURE);
  }

  //-----------------------------
  srand(time(NULL));
  int const blocks = 128;
  int const threads = 128;
  //size_t size = blocks*threads*16; // 一共多少字节;每个线程16个数值？
  size_t size = blocks*threads; //猜测是例子写错了·


  auto lambdaHost = [](half2*p){cudaFreeHost(p);};
  auto lambdaDev = [](half2*p){cudaFree(p);};
  // 申请输入的内存 
  vector<unique_ptr<half2,void(*)(half2*)>> vec;
  vector<unique_ptr<half2,void(*)(half2*)>> devVec;
  //size 本身就包含了16bit了
  for(int i=0;i<2;i++){
    half2* tmp;
    checkCudaErrors(cudaMallocHost((void**)&tmp,size*sizeof(half2)));
    vec.emplace_back(tmp,lambdaHost);

    half2* tmp1;
    checkCudaErrors(cudaMalloc((void**)&tmp1, size*sizeof(half2) ));
    devVec.emplace_back(tmp1,lambdaDev);
  }
  //申请输出的内存
  unique_ptr<float,void(*)(float*)> results{nullptr,[](float*p){cudaFreeHost(p);}};
  unique_ptr<float,void(*)(float*)> devResults{nullptr,[](float*p){cudaFree(p);}};
  float* _results;
  float* _devResults;
  checkCudaErrors(cudaMallocHost((void**)&_results,blocks*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&_devResults,blocks*sizeof(float)));
  results.reset(_results);
  devResults.reset(_devResults);

  //执行核操作
  for(int i=0; i<2;i++){
    genInput(vec[i].get(),size);
    checkCudaErrors(cudaMemcpy(devVec[i].get(),vec[i].get(),
                               size*sizeof(half2),
                               cudaMemcpyHostToDevice));

    scalarProductKernel<<<blocks, threads>>>(devVec[0].get(),
                                             devVec[1].get(),
                                             devResults.get(), size);
  }
  checkCudaErrors(cudaMemcpy(results.get(), devResults.get(), blocks*sizeof(half2),
                             cudaMemcpyDeviceToHost));

  float res = 0;
  for(int i=0; i<blocks; i++){
    res += *(results.get()+i);
  }
  printf("Result: %f \n", res);

  exit(EXIT_SUCCESS);
}

