#ifndef _SHAREDMEM_H_
#define _SHAREDMEM_H_

//------------------------------------------------------
// 因为动态size shared memory 需要先声明成extern，
// 而我们不能直接对其进行模板化。为了达到目的，先声明
// 一个简单的struct对其封装，用不同的类型定义的名称来声明
// 这个extern 数组以此避免编译器的重复定义
//
// 为了在模板化的__global__ 或者__device__函数中，
// 使用动态分配的shared memory, 只需要简单的如下替换 
//
//template<typename T> __global__ void
//foo(T* g_idata, T* g_odata){
//  extern __shared__ T sdata[];
//  ...
//  doStuff(sdata);
//  ...
//}
//替换成
//template<typename T>__global__ void
//foo(T *g_idata, T *g_odata){
//  SharedMemory<T> smem;
//  T *sdata = smem.getPointer();
//  ...
//  doStuff(sdata);
//  ...
//}
//------------------------------------------------------

// 首先进行模板化，然后进行特例化
// 这是一个未特例化的模板，
// 通过放置一个未定义的符号在函数体内来阻止编译器对其进行生成
// 如果程序中用到主模板的话，需要在某处给出定义，但是如果主模板在程序中
// 从未实例化，则无须定义
template<typename T>
struct SharedMemory{
  
  __device__ T *getPointer(){
    extern __device__ void error(void); 
    error();
    return nullptr;
  }
};

// 模板特例化

template<>
struct SharedMemory<int>{
  __device__ int *getPointer(){
    extern __shared__ int s_int[];
    return s_int;
  }
};

template<>
struct SharedMemory<unsigned int>{
  __device__ unsigned int *getPointer(){
    extern __shared__ unsigned int s_uint[];
    return s_uint;
  }
};

template<>
struct SharedMemory<char>{
  __device__ char *getPointer(){
    extern __shared__ char s_char[];
    return s_char;
  }
};

template<>
struct SharedMemory<unsigned char>{
  __device__ unsigned char *getPointer(){
    extern __shared__ unsigned char s_uchar[];
    return s_uchar;
  }
};

template<>
struct SharedMemory<short>{
  __device__ short *getPointer(){
    extern __shared__ short s_short[];
    return s_short;
  }
};

template<>
struct SharedMemory<unsigned short>{
  __device__ unsigned short *getPointer(){
    extern __shared__ unsigned short s_ushort[];
    return s_ushort;
  }
};

template<>
struct SharedMemory<long>{
  __device__ long *getPointer(){
    extern __shared__ long s_long[];
    return s_long;
  }
};

template<>
struct SharedMemory<unsigned long>{
  __device__ unsigned long *getPointer(){
    extern __shared__ unsigned long s_ulong[];
    return s_ulong;
  }
};

template<>
struct SharedMemory<bool>{
  __device__ bool *getPointer(){
    extern __shared__ bool s_bool[];
    return s_bool;
  }
};

template<>
struct SharedMemory<float>{
  __device__ float *getPointer(){
    extern __shared__ float s_float[];
    return s_float;
  }
};

template<>
struct SharedMemory<double>{
  __device__ double *getPointer(){
    extern __shared__ double s_double[];
    return s_double;
  }
};

#endif
