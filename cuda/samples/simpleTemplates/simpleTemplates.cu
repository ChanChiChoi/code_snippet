#include<cstdio>
#include<iostream>
#include<cassert>
#include<cstring>
#include<cmath>
#include<chrono>
#include<fstream>

#include<helper_cuda.h>
#include<timer.h>

// 当在main中只有一次实例化runTest时，之前写法是没问题的
// 但是当实例化超过2回，编译器内部其实就需要创建两次定义，
//*****************************************************
// #include<cstdio>
// template<typename T>
// __global__ void
// test(){
//   extern __shared__ T data[];
//   printf("inside\n");
// }
// 
// int main(){
//   test<int><<<3,4,5>>>();
//   test<double><<<3,4,5>>>();
//   cudaDeviceSynchronize();
// }
//*****************************************************
// 但是extern是外部的意思，从而会造成冲突
// 所以需要采用诸如sharedmem.cuh中的进行封装

#include "sharedmem.cuh"

using std::cout;
using std::endl;
using std::cerr;
using std::chrono::system_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

template<typename T> __global__ void
testKernel(T *g_idata, T *g_odata){

  SharedMemory<T> smem;
  T *sdata = smem.getPointer();
//  extern __shared__ T sdata[];

  unsigned int const tid = threadIdx.x;
  unsigned int const num_threads = blockDim.x;
  
  //global -> shared
  sdata[tid] = g_idata[tid];
  __syncthreads();
  
  //
  sdata[tid] = (T)num_threads * sdata[tid];
  __syncthreads();
  //shared -> global
  g_odata[tid] = sdata[tid];
}

//---------------------------
template<typename T> void
computeGold(T *ref, T *idata, unsigned int const len){
  T const T_len = static_cast<T>(len); // 编译时刻转换
  for(unsigned int i=0; i<len; i++)
    ref[i] = idata[i]*T_len;
}

//----------------------------
template<typename T, typename S> inline bool
compareData(T const *ref, T const *data, unsigned int const len,
           S const epsilon, float const threshold){

    assert(epsilon >= 0);
    bool result = true;
    unsigned int error_count = 0;
    for(unsigned int i=0; i<len; i++){
      float diff = (float)ref[i] - (float)data[i];
      bool comp = (diff <= epsilon) && (diff >= -epsilon);
      result &= comp;
      error_count += !comp;
    }
    if (threshold == 0.0f)
      return result ? true: false;
    else{
      if(error_count)
        printf("%4.2f(%%) of bytes mismatched (count=%d)\n",
               (float)error_count*100/(float)len, error_count);
      return (len*threshold > error_count)? true: false;
    }
}
//-----------------------------
template<typename T>
class ArrayComparator{
  public:
    bool compare(T const *ref, T *data, unsigned int len){
      fprintf(stderr,"Error: 没有为此类型实现的对比函数\n");
      return false;
    }
};

template<>
class ArrayComparator<int>{
  public:
    bool compare(int const *ref, int *data, unsigned int len){
      return compareData(ref, data, len, 0.15f, 0.0f);
    }
};

template<>
class ArrayComparator<float>{
  public:
    bool compare(float const *ref, float *data, unsigned int len){
      return compareData(ref, data, len, 0.15f, 0.0f);
    }
};
//--------------------------
template<typename T, typename S> inline bool
writeFile(char const *filename, T const *data, unsigned int len,
          S const epsilon, bool verbose, bool append=false){
  assert(nullptr != filename);
  assert(nullptr != data);

  std::fstream fh(filename, std::fstream::out| std::fstream::ate);
  if(verbose)
    cerr<<"writeFile: open file "<<filename<<" for write/append."<<endl;
  if(!fh.good()){
    return false;
  }
  fh<<"# "<<epsilon<<endl;
  //write data
  for(unsigned int i=0; (i<len)&&(fh.good()); i++)
    fh<<data[i]<<' ';

  if(!fh.good()){
    if(verbose)
     cerr<<"writeFile: writing file failed"<<endl;
     return false;
  }
  fh<<endl;
  return true;
}
//---------------------------
template<typename T>
class ArrayFileWriter{
  public:
    bool write(char const *filename, T *data, unsigned int len, float epsilon){
       fprintf(stderr, "Error: 没有为此类型实现write 函数\n");
       return false;
    }
};

template<>
class ArrayFileWriter<int>{
  public:
    bool write(char const *filename, int *data, unsigned int len, float epsilon){
      return writeFile(filename, data, len, epsilon, false);
    }
};

template<>
class ArrayFileWriter<float>{
  public:
    bool write(char const *filename, float *data, unsigned int len, float epsilon){
      return writeFile(filename, data, len, epsilon, false);
    }
};
//----------------------------
template<typename T> void
runTest(int argc, char *argv[], int len, int g_TotalFailres){
  
  unsigned int num_threads = len;
  unsigned int mem_size = sizeof(float)*num_threads;

  auto st = system_clock::now();

  // 分配host内存并初始化
  T *h_idata = (T*)malloc(mem_size);
  for(unsigned int i=0; i<num_threads; i++){
    h_idata[i] = (T)i;
  }  

  //分配device的输入内存并复制
  T *d_idata;
  checkCudaErrors(cudaMalloc((void**)&d_idata, mem_size));
  checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));
  //分配device的输出内存
  T *d_odata;
  checkCudaErrors(cudaMalloc((void**)&d_odata, mem_size));
  // kernel调度
  dim3 grid(1,1,1);
  dim3 threads(num_threads,1,1);

  testKernel<T><<<grid,threads,mem_size>>>(d_idata, d_odata);

  getLastCudaError("Kernel execution failed");
  // 分配host侧的结果内存,然后结果cp到host
  T *h_odata = (T*)malloc(mem_size); 
  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(T)*num_threads,
                                cudaMemcpyDeviceToHost));
  auto ed = system_clock::now();
  cout<<"take "<<duration_cast<milliseconds>(ed-st).count()<<" ms"<<endl;

  T *ref = (T*)malloc(mem_size);
  computeGold<T>(ref, h_idata, num_threads);
  ArrayComparator<T> comparator;
  ArrayFileWriter<T> writer;

  bool isRegression = false;
  if(isRegression)
    writer.write("./data/regression.dat", h_odata, num_threads, 0.0f);
  else{
    bool res = comparator.compare(ref, h_odata, num_threads);
    cout<<"Compare "<<((1 == res) ? "OK": "MISMATCH")<<endl;
    g_TotalFailres += (1 != res);
  }

  free(h_idata);
  free(h_odata);
  free(ref);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));
  
  
}


int
main(int argc, char *argv[]){

  int g_TotalFailres=0;
  cout<<"> runTest<float, 32>"<<endl;
  runTest<float>(argc, argv, 32,g_TotalFailres);
  cout<<"> runTest<int, 64>"<<endl;
  runTest<int>(argc, argv, 64, g_TotalFailres);

  cout<<"[simpleTemplates] -> Test Results: "<<g_TotalFailres<<" Failures"<<endl;
  exit(g_TotalFailres == 0? EXIT_SUCCESS: EXIT_FAILURE);
}

//需要使用模板特例化来解决针对不同类型下CUTIL的数组对比和文件写功能
