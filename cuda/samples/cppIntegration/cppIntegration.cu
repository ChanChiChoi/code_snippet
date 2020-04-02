#include<cmath>
#include<cassert>
#include<cstdio>
#include<memory>

#include<helper_cuda.h>
//#include<cuda_runtime.h>
#include<vector_types.h>


using namespace std;

//#ifndef MAX
//#define MAX(a,b) (a>b ?a:b)
//#endif

//在cpp文件中定义，并且在cpp中输出的时候也需要c风格形式
extern "C" void
computeGold(std::unique_ptr<char[]>&ref, char *idata, unsigned int const len);

extern "C" void
computeGold2(std::unique_ptr<int2[]>&ref, int2 *idata, unsigned int const len);

__global__ void 
kernel(int *g_data){

  unsigned int const tid = threadIdx.x;
  //之前是char类型，这里是int类型，也就是一次读取4个char
  int data = g_data[tid];
/*
 use integer arithmetic to process all four bytes with one thread
 this serializes the execution, but is the simplest solutions to avoid
 bank conflicts for this very low number of threads
 in general it is more efficient to process each byte by a separate thread,
 to avoid bank conflicts the access pattern should be
 g_data[4 * wtid + wid], where wtid is the **thread id within the half warp**
 and **wid is the warp id**
 see also the programming guide for a more in depth discussion.
*/
 //又不是从shared部分取数据，哪门子的bank conflict啊？
  //提取从左到右的8bit，(data<<x)>>24; 
  g_data[tid] = ( (data<<0)>>24 - 10 )<<24 |
                ( (data<<8)>>24 - 10 )<<16 |
                ( (data<<16)>>24 - 10)<<8 |
                ( (data<24)>>24 - 10)<<0 ;
}

__global__ void
kernel2(int2 *g_data){
  unsigned int const tid = threadIdx.x;
  int2 data = g_data[tid];
  //这里又bb一通上面kernel里面的话的，都怀疑是复制过来的
  g_data[tid].x = data.x - data.y;
}


extern "C" bool
runTest(int const argc, char const *argv[], char *data,
        int2 *data_int2, unsigned int len){

  //检测是否刚好是4的倍数
  assert(0 == (len%4) );
  unsigned int const num_threads = len / 4;
  unsigned int const mem_size = sizeof(char)*len;
  unsigned int const mem_size_int2 = sizeof(int2)*len;

  //device mem
  //TODO:设计一个类似unique_ptr的模板，然后来封装cudaMalloc
  char *d_data;
  checkCudaErrors(cudaMalloc((void**)&d_data,mem_size));
  checkCudaErrors(cudaMemcpy(d_data, data, mem_size, cudaMemcpyHostToDevice));

  //for int2 version
  int2 *d_data_int2;
  checkCudaErrors(cudaMalloc((void**)&d_data_int2, mem_size_int2));
  checkCudaErrors(cudaMemcpy(d_data_int2, data_int2, 
                             mem_size_int2,cudaMemcpyHostToDevice));

  dim3 grids(1,1,1);
  dim3 blocks(num_threads,1,1);//对于int类型，4个char一个线程;16/4
  dim3 blocks2(len,1,1);//对于int2类型，2个int 一个线程;16

  kernel<<<grids,blocks>>>(reinterpret_cast<int *>(d_data));
  kernel2<<<grids,blocks2>>>(d_data_int2);

  //TODO:
  // 不知道为什么下面这句话报错
  //checkCudaErrors(getLastCudaError("kernel execution failed"));
  //getLastCudaError("kernel execution failed");
  checkCudaErrors(cudaDeviceSynchronize());

  unique_ptr<char[]> ref{new char[mem_size]};  
  computeGold(ref, data, len);

  unique_ptr<int2[]> ref2{new int2[mem_size_int2]};
  computeGold2(ref2, data_int2, len);

  checkCudaErrors(cudaMemcpy(data,d_data,mem_size,cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(data_int2, d_data_int2,mem_size_int2,
                             cudaMemcpyDeviceToHost));

  bool flag = true;
  
  for(unsigned int i=0; i<len; i++){
    if(ref[i] != data[i] ||
       ref2[i].x != data_int2[i].x ||
       ref2[i].y != data_int2[i].y)

       flag = false;
  }

  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_data_int2));

  return flag;
}
