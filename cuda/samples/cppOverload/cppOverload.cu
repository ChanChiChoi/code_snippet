#include<iostream>
#include<memory>

#include<helper_cuda.h>

using namespace std;

#define N 1024
#define NUM_THREADS 256
#define DIV_UP(a,b) (((a)+(b)-1)/(b))

#include"kernel_overload.cuh"

typedef void(*fp)(int*);

int
_main(int argc, char const *argv[]){

  cout<<argv[0]<<" Starting..."<<endl;

  int devID = 0;
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop,devID));
  if(prop.major<2){
    cerr<<"ERROR: cppOverload requireds sm 2.0 or higher"<<endl;
    cout<<"Current GPU device has SM "<<prop.major<<"."<<prop.minor<<" Exiting..."<<endl;
    exit(EXIT_FAILURE);
  }
  checkCudaErrors(cudaSetDevice(devID));

  //分配内存
  int *_dInput;
  int *_dOutput;
  int *_hInput;
  int *_hOutput;
  checkCudaErrors(cudaMalloc((void**)&_dInput, 2*N*sizeof(int) ) );
  checkCudaErrors(cudaMalloc(&_dOutput, sizeof(int)*N));
  unique_ptr<int,fp>dInput{_dInput, [](int*p){cudaFree(p);} };
  unique_ptr<int,fp>dOutput{_dOutput, [](int*p){cudaFree(p);}};

  checkCudaErrors(cudaMallocHost(&_hInput, sizeof(int)*N*2));
  checkCudaErrors(cudaMallocHost(&_hOutput, sizeof(int)*N));
  unique_ptr<int,fp>hInput{_hInput, [](int*p){cudaFreeHost(p);}};
  unique_ptr<int,fp>hOutput{_hOutput, [](int*p){cudaFreeHost(p);}};

  for(int i=0;i<N*2; i++){
    *(hInput.get()+i) = i;
  }
  checkCudaErrors(cudaMemcpy(dInput.get(),hInput.get(),
                             sizeof(int)*N*2, cudaMemcpyHostToDevice ));

  //测试cpp重载
  int a = 1;
  
  simple_kernel<<< DIV_UP(N,NUM_THREADS), NUM_THREADS>>>(dInput.get(), dOutput.get(), a);
  checkCudaErrors(cudaMemcpy(hOutput.get(), dOutput.get(), sizeof(int)*N, cudaMemcpyDeviceToHost));

  simple_kernel<<<DIV_UP(N,NUM_THREADS),NUM_THREADS>>>(reinterpret_cast<int2*>(dInput.get()), dOutput.get(),a);
  checkCudaErrors(cudaMemcpy(hOutput.get(), dOutput.get(), sizeof(int)*N, cudaMemcpyDeviceToHost));

  simple_kernel<<<DIV_UP(N,NUM_THREADS),NUM_THREADS>>>(dInput.get(),dInput.get()+N, dOutput.get(),a);
  checkCudaErrors(cudaMemcpy(hOutput.get(), dOutput.get(), sizeof(int)*N, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaDeviceSynchronize());
//  cudaFree(_dOutput);
//  cudaFreeHost(_hInput);
//  cudaFreeHost(_hOutput);
  return 0;
}

int 
main(int argc, char const*argv[]){
  _main(argc, argv);
  checkCudaErrors(cudaDeviceReset());
  
}
